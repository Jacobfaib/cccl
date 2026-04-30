//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_TRANSFORM_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_TRANSFORM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__multi_gpu/sharded_container.h>
#include <cuda/experimental/__multi_gpu/thread_group.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <typename InputIt, typename OutputIt, typename F>
void transform(
  const thread_group& group, const basic_sharded_buffer<InputIt>& input, basic_sharded_buffer<OutputIt>& output, F&& op)
{
  const auto input_shards  = input.shards();
  const auto output_shards = output.shards();

  group.dispatch(input_shards.size(), [&](::cuda::std::uint32_t i) {
    auto&& i_shard = input_shards[i];
    auto&& o_shard = output_shards[i];

    const auto _ = __ensure_current_context{i_shard.device().context()};

    thrust::transform(thrust::device, i_shard.iter_begin(), i_shard.iter_end(), o_shard.iter_begin(), op);
  });
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___THREAD_GROUP_THREAD_GROUP_H
