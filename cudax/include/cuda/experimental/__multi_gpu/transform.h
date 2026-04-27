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

#include <thrust/transform.h>

#include <cuda/std/cstdint>

#include <cuda/experimental/__multi_gpu/thread_group.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <typename InputIt, typename OutputIt, typename F>
void transform(const thread_group& group, InputIt first1, InputIt last1, OutputIt d_first, F&& op)
{
  auto input_shards  = first1.shards();
  auto output_shards = d_first.shards();

  group.dispatch(input_shards.size(), [&](::cuda::std::uint32_t i) {
    auto&& i_shard = input_shards[i];
    auto&& o_shard = output_shards[i];

    thrust::transform(i_shard.begin(), i_shard.end(), o_shard.begin(), op);
  });
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___THREAD_GROUP_THREAD_GROUP_H
