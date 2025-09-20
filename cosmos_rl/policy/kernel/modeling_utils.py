# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from flash_attn import (
    flash_attn_func as ori_flash_attn_func,
    flash_attn_varlen_func as ori_flash_attn_varlen_func,
)
from flash_attn.layers.rotary import apply_rotary_emb as ori_apply_rotary_emb
import torch

def _lastdim_contig(x: torch.Tensor) -> torch.Tensor:
    return x if x is None or x.stride(-1) == 1 else x.contiguous()

def _flash_attn_func(q, k, v, *args, **kwargs):
    q = _lastdim_contig(q)
    k = _lastdim_contig(k)
    v = _lastdim_contig(v)
    return ori_flash_attn_func(q, k, v, *args, **kwargs)

def _flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *args, **kwargs):
    q = _lastdim_contig(q)
    k = _lastdim_contig(k)
    v = _lastdim_contig(v)
    cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
    cu_seqlens_k = cu_seqlens_k.to(torch.int32).contiguous()
    return ori_flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *args, **kwargs
    )

def _apply_rotary_emb(*args, **kwargs):
    return ori_apply_rotary_emb(*args, **kwargs)

flash_attn_func = _flash_attn_func
flash_attn_varlen_func = _flash_attn_varlen_func
apply_rotary_emb = _apply_rotary_emb

def set_flash_attn_deterministic(deterministic: bool):
    global flash_attn_func, flash_attn_varlen_func, apply_rotary_emb
    if deterministic:
        flash_attn_func = partial(_flash_attn_func, deterministic=True)
        flash_attn_varlen_func = partial(_flash_attn_varlen_func, deterministic=True)
    else:
        flash_attn_func = _flash_attn_func
        flash_attn_varlen_func = _flash_attn_varlen_func
