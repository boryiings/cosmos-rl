from typing import Dict, Tuple, Optional

import torch
from torch.nn import Parameter

import transformer_engine as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

from vllm.model_executor.layers.quantization.utils.w8a8_utils import Fp8LinearOp
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import w8a8_utils

from cosmos_rl.policy.model import WeightMapper
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.logging import logger

"""
This file is used to patch the vllm model to use rowwise fp8 linear.
"""
class QuantizedParameter(nn.Parameter):
    """
    A custom parameter that holds quantization metadata.
    """
    def __new__(cls, data=None, requires_grad=True, scale=None, zero_point=None):
        # Create the underlying nn.Parameter tensor object
        if data is None:
            data = torch.empty(0)
        
        # This calls the __new__ method of the parent class (nn.Parameter)
        # to create the actual tensor object.
        instance = super().__new__(cls, data._columnwise_data, requires_grad=requires_grad)

        # Attach custom attributes to the new instance
        instance.nvfp4_type = data
        
        return instance

    def __repr__(self):
        # Customize the string representation to show our metadata
        return (f"QuantizedParameter(scale={self.scale}, zero_point={self.zero_point}, "
                f"data=\n{super().__repr__()})")


def simplify_process_weights_after_loading():
    """
    This function is used to simplify the process_weights_after_loading of Fp8LinearMethod in vLLM, to quantize the
    weight of linear only in `rowwise` mode.
    Refer to the method `process_weights_after_loading`:
    https://github.com/vllm-project/vllm/blob/1a4f35e2eaa3ebdecb8ef9ff8302b01e289305c9/vllm/model_executor/layers/quantization/fp8.py#L319
    """

    def simplified_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_shape = layer.weight.shape
        te_dtype = tex.DType.kFloat4E2M1
        w_quantizer = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=False, columnwise=True)
        w_nvfp4 = w_quantizer.make_empty(w_shape, dtype=layer.weight.dtype)
        w_nvfp4 = w_quantizer.update_quantized(layer.weight, w_nvfp4)

        # Update the layer with the new values
        layer.weight = QuantizedParameter(w_nvfp4)
        layer.weight_scale = None
        layer.input_scale = None

    # modify the process_weights_after_loading method for rowwise fp8 linear.
    Fp8LinearMethod.process_weights_after_loading = (
        simplified_process_weights_after_loading
    )


simplify_process_weights_after_loading()


class NVFP4LinearOp:
    def __init__(self):
        te_dtype = tex.DType.kFloat4E2M1
        self.x_quantizer = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=False)
        self.workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)


    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: Optional[torch.dtype] = None,
        input_scale: Optional[torch.Tensor] = None,
        input_scale_ub: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Quantize the input tensor to NVFP4
        x_shape = input.shape
        te_dtype = tex.DType.kFloat4E2M1
        x_nvfp4 = self.x_quantizer.make_empty(x_shape, dtype=input.dtype)
        x_nvfp4 = self.x_quantizer.update_quantized(input, x_nvfp4)

        # All other GEMM args
        out_quantizer, bias, gelu_input, D_preallocated = None, None, None, None
        bias_dtype = TE_DType[torch.bfloat16]
        use_gelu, use_grad, accumulate, use_split_accumulator = False, False, False, False
        workspace_size = self.workspace.numel()
        transa, transb = False, False
        return tex.generic_gemm(
            weight.nvfp4_type,
            transa,
            x_nvfp4,
            transb,
            D_preallocated,
            out_quantizer,
            TE_DType[torch.bfloat16],
            bias,
            bias_dtype,
            use_gelu,
            gelu_input,
            use_grad,
            self.workspace,
            workspace_size,
            accumulate,
            use_split_accumulator,
        )[0]


# patch the Linear layer.
def apply_fp8_linear_patch(model: torch.nn.Module):
    for name, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        logger.info(f"Found module name {name}, quant_method = {quant_method}")
        if quant_method is None:
            continue
        elif isinstance(quant_method, Fp8LinearMethod):
            # replace the fp8_linear op with our own config
            # that use rowwise fp8
            # WARNING: in `Fp8LinearOp` `__init__`, vllm will read the `vllm_config`
            # But at this time, `vllm_config` is empty. So there will have a warning that complains
            # it is not set. This only affects the padding, seems not a big problem.
            quant_method.fp8_linear = NVFP4LinearOp()
            logger.info(f"Found Fp8LinearMethod, name = {name}, quant_method = {quant_method}")
        else:
            # We will not handle other quant methods.
            pass


def replace_weight_of_quantized_module(
    vllm_model: torch.nn.Module,
    cached_weight_map: Dict[str, torch.Tensor],
    weight_mapper: WeightMapper,
):
    """
    Temporarily replace the quantized fp8 layer's weight with the cached weight.
    """
    for name, module in vllm_model.named_modules():
        # Here we use the compatible name as the key, aligned with what we do in
        # `cache_weight_of_quantized_module` and `rollout_prepare_recv`.
        compatible_name = weight_mapper._rollout_vllm_name_to_hf(name + ".weight")
        if compatible_name in cached_weight_map:
            module.weight = cached_weight_map[compatible_name]


def cache_weight_of_quantized_module(
    vllm_model: torch.nn.Module,
    promotion_dtype: torch.dtype,
    weight_mapper: WeightMapper,
    parallel_dims: ParallelDims,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Get the weight from the quantized module."""
    original_weight_map = {}
    hp_weight_map = {}
    for name, module in vllm_model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None:
            continue
        elif isinstance(quant_method, Fp8LinearMethod):
            weight_name = name + ".weight"
            compatible_name = weight_mapper._rollout_vllm_name_to_hf(weight_name)
            original_weight_map[compatible_name] = (
                module.weight
            )  # qweight has shape [in_dim, out_dim]
            hp_weight = (
                module.weight.to(promotion_dtype).t().contiguous()
            )  # hp weight has shape [out_dim, in_dim]
            hp_weight_map[compatible_name] = Parameter(hp_weight, requires_grad=False)
        else:
            # We will not handle other quant methods.
            pass
    return hp_weight_map, original_weight_map


def post_process_view_map_for_fp8(
    vllm_weight_inplace_view_map: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Process the view map returned by `rollout_prepare_recv`.
            - remove the weight_scale from the view map.
    Args:
        vllm_weight_inplace_view_map (Dict[str, torch.Tensor]): view map returned by `rollout_prepare_recv`
    Returns:
        Dict[str, torch.Tensor]: view map doesn't contain weight_scale.
    """
    processed_view_map = {}
    for key, value in vllm_weight_inplace_view_map.items():
        if "weight_scale" in key:
            continue
        processed_view_map[key] = value
    return processed_view_map
