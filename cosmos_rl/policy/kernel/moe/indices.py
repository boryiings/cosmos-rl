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

import torch
import triton
import triton.language as tl


__all__ = ["generate_permute_indices"]


# parallelized kernel
@triton.jit
def _fill_indices_kernel(
    tokens_per_expert_group_ptr,
    start_index_values_ptr,
    write_offsets_ptr,
    output_ptr,
    experts_per_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # Number of threads per block
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    # map programs (blocks) to the experts and loop (grid stride) if needed
    for expert_id in range(pid, experts_per_rank, num_programs):
        # read this experts write offset
        write_offset = tl.load(write_offsets_ptr + expert_id)

        for r in range(num_ranks):
            # index into tokens_per_expert_group array
            i = r * experts_per_rank + expert_id

            # load start index and number of tokens for this expert-rank pair
            start_index = tl.load(start_index_values_ptr + i)
            length = tl.load(tokens_per_expert_group_ptr + i)

            # each thread in block processes tokens in parallel
            offsets = tl.arange(0, BLOCK_SIZE)

            # tokens are processed in chunks of BLOCK_SIZE
            for chunk_start in range(0, length, BLOCK_SIZE):
                chunk_offsets = chunk_start + offsets

                # mask valid indices
                mask = chunk_offsets < length

                values = start_index + chunk_offsets

                # destination
                dest_indices = write_offset + chunk_offsets

                # store
                tl.store(output_ptr + dest_indices, values, mask=mask)

            # update write offset for next rank
            write_offset += length


# ==============
# wrapper
# ==============


def fill_indices_wrapper(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    total_size: int,
    block_size: int = 128,
    max_blocks: int = 1024,
):
    # Allocate exact size needed instead of max_len
    permuted_indices = torch.full(
        (total_size,), -1, dtype=torch.int32, device=tokens_per_expert_group.device
    )

    # write offsets is per local expert...
    num_blocks = min(experts_per_rank, max_blocks)
    # grid = one block per expert unless capped and then we loop...
    grid = (num_blocks,)

    # launch kernel
    _fill_indices_kernel[grid](
        tokens_per_expert_group,
        start_index_values,
        write_offsets,
        permuted_indices,
        experts_per_rank,
        num_ranks,
        BLOCK_SIZE=block_size,
    )
    return permuted_indices


# used for reference testing only


def fill_indices_cpu(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    total_size: int,  # Changed from max_len to actual required size
):
    # Allocate exact size needed
    permuted_indices = torch.full(
        (total_size,),
        -1,
        dtype=torch.int32,
    )

    # Fill the permuted indices
    for e in range(experts_per_rank):
        write_start = write_offsets[e].item()
        for r in range(num_ranks):
            i = r * experts_per_rank + e
            start_index = start_index_values[i].item()
            length = tokens_per_expert_group[i].item()
            if length > 0:
                end_idx = min(write_start + length, total_size)
                permuted_indices[write_start:end_idx] = torch.arange(
                    start_index,
                    start_index + (end_idx - write_start),
                    dtype=torch.int32,
                )
            write_start += length
    return permuted_indices


def generate_permute_indices(
    tokens_per_expert_group: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    alignment: int,
    use_cpu: bool = False,
):
    """
    Prepare permutation indices and the number of tokens for each expert.
    Modified version that returns a tensor of size sum(m_sizes) instead of max_len.

    Args:
        tokens_per_expert_group: number of tokens for each expert from all ranks.
        experts_per_rank: number of experts per rank.
        num_ranks: number of ranks.
        alignment: alignment for each returned element in `m_sizes` and padding min for zero token experts.
        use_cpu: whether to use CPU implementation.

    Returns:
        permuted_indices: Tensor of indices with size sum(m_sizes), that map original token order to the expert-grouped order.
        m_sizes: aligned number of tokens for each expert (padded to alignment boundary).
        m_offsets: Cumulative sum of m_sizes. The exclusive ending position for each expert's tokens.

    Explanatory details:
        `tokens_per_expert_group` is of shape (num_ranks * experts_per_rank,), for example:
        From: |       rank 0      |       rank 1      |
        To:   | E0 | E1 | E2 | E3 | E0 | E1 | E2 | E3 |
              |  4 |  2 |  1 |  3 |  1 |  2 |  3 |  4 |
    """

    # prefix sum to get start index of each expert
    start_index_values = (
        torch.cumsum(tokens_per_expert_group, 0) - tokens_per_expert_group
    )

    # total tokens for each expert (sum over ranks)
    total_tokens_per_expert = tokens_per_expert_group.view(num_ranks, -1).sum(0)

    # pad out empty experts to alignment requirement
    total_tokens_per_expert = torch.clamp_min(total_tokens_per_expert, alignment)

    # align the chunk sizes (cdiv)
    m_sizes = ((total_tokens_per_expert + alignment - 1) // alignment * alignment).to(
        torch.int32
    )

    # additional prefix sum to get write offset of each expert in permuted_indices
    m_offsets = torch.cumsum(m_sizes, 0)
    write_offsets = m_offsets - m_sizes

    # Calculate the actual total size needed
    total_size = m_offsets[-1]

    # Select the implementation to use
    if use_cpu:
        permuted_indices = fill_indices_cpu(
            tokens_per_expert_group,
            start_index_values,
            write_offsets,
            experts_per_rank,
            num_ranks,
            total_size,
        )
    else:  # gpu
        permuted_indices = fill_indices_wrapper(
            tokens_per_expert_group,
            start_index_values,
            write_offsets,
            experts_per_rank,
            num_ranks,
            total_size,
        )

    return permuted_indices, m_sizes, m_offsets.to(torch.int32)


# Below is for testing only


def simple_test():
    device = torch.device("cuda", 0)
    experts_per_rank = 4
    num_ranks = 4
    tokens_per_expert_group = torch.full(
        (num_ranks * experts_per_rank,), 4, dtype=torch.int32, device=device
    )
    alignment = 32
    # Use the GPU kernel
    permuted_indices_gpu, m_sizes, _ = generate_permute_indices(
        tokens_per_expert_group,
        experts_per_rank,
        num_ranks,
        alignment,
        use_cpu=False,
    )
    # Use the CPU method
    permuted_indices_cpu, m_sizes, _ = generate_permute_indices(
        tokens_per_expert_group,
        experts_per_rank,
        num_ranks,
        alignment,
        use_cpu=True,
    )
    # Check that the results are the same

    assert torch.equal(permuted_indices_gpu.cpu(), permuted_indices_cpu)
    assert torch.equal(
        torch.remainder(m_sizes, alignment),
        torch.zeros(experts_per_rank, device=device),
    )
    # Print the results
    print(f"{permuted_indices_gpu=}, \n{permuted_indices_cpu=}")
    print(f"{m_sizes=}")
    print("Success")
    return True  # assert would have failed meaning getting here is success.


def test_with_zero_tokens():
    device = torch.device("cuda", 0)
    experts_per_rank = 4
    num_ranks = 2

    # Create a test case where some experts have zero tokens
    tokens_per_expert_group = torch.tensor(
        [4, 0, 2, 3, 1, 0, 0, 5],  # Some experts have zero tokens
        dtype=torch.int32,
        device=device,
    )

    alignment = 8

    # Use the GPU kernel
    permuted_indices_gpu, m_sizes, m_offsets = generate_permute_indices(
        tokens_per_expert_group,
        experts_per_rank,
        num_ranks,
        alignment,
        use_cpu=False,
    )

    # Use the CPU method
    permuted_indices_cpu, m_sizes_cpu, m_offsets_cpu = generate_permute_indices(
        tokens_per_expert_group,
        experts_per_rank,
        num_ranks,
        alignment,
        use_cpu=True,
    )

    # Check that the results are the same
    assert torch.equal(permuted_indices_gpu.cpu(), permuted_indices_cpu)
    assert torch.equal(m_sizes, m_sizes_cpu)

    # Verify that experts with zero tokens have at least min_slots_per_expert
    total_tokens_per_expert = tokens_per_expert_group.view(num_ranks, -1).sum(0)
    zero_token_experts = total_tokens_per_expert == 0
    if zero_token_experts.any():
        assert (m_sizes[zero_token_experts] >= alignment).all()

    # Check alignment
    assert torch.equal(
        torch.remainder(m_sizes, alignment),
        torch.zeros(experts_per_rank, device=device),
    )

    # Print the results
    print(f"tokens_per_expert_group = {tokens_per_expert_group}")
    print(f"total_tokens_per_expert = {total_tokens_per_expert}")
    print(f"m_sizes = {m_sizes}")
    print(f"m_offsets = {m_offsets}")
    print(f"permuted_indices = {permuted_indices_gpu[:sum(m_sizes).item()]}")

    # Check that experts with zero tokens have -1 in their slots
    for e in range(experts_per_rank):
        start = (m_offsets[e] - m_sizes[e]).item()
        end = m_offsets[e].item()
        expert_indices = permuted_indices_gpu[start:end]
        if total_tokens_per_expert[e] == 0:
            assert (
                expert_indices == -1
            ).all(), f"Expert {e} with zero tokens should have all -1 indices"
            assert (
                expert_indices.size(0) >= alignment
            ), f"Expert {e} with zero tokens should have at least {alignment} slots"
            print(
                f"Expert {e} has zero tokens and {expert_indices.size(0)} slots with all -1"
            )

    print("All tests passed successfully!")
    return True


if __name__ == "__main__":
    simple_test()
    test_with_zero_tokens()
