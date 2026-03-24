//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include <cuda/experimental/__sharded_mdspan/sharded_mdspan.hpp>
#include <cuda/experimental/__sharded_mdspan/thread_group.hpp>

#include <cstddef>
#include <utility>

namespace cuda::experimental
{
template <typename MDSpan, typename F>
void transform(const thread_group& group,
               const basic_sharded_mdspan<MDSpan>& in_mdspan,
               const basic_sharded_mdspan<MDSpan>& out_mdspan,
               F&& functor)
{
  for (auto&& device : group.devices())
  {
    thrust::transform(in_shard_it->subspan.data(), out_shard_it->subspan.data(), ::std::forward<F>(functor));
  }
}

inline void foo()
{
  auto v1 = thrust::host_vector<int>{4};
  auto v2 = thrust::host_vector<int>{4};

  auto v1_span = ::cuda::std::mdspan{v1.data(), 1, 2};
  auto v2_span = ::cuda::host_mdspan{v2.data(), 4, 5};

  auto sharded = basic_sharded_mdarray{{v1_span, logical_device{0}}, {v2_span, host_memory{}}};

  auto group = host_thread_group{0, 4};

  transform(group, sharded, sharded, [](int v) {
    return v + 1;
  });

  transform(group, sharded, sharded, [subspan = sharded.shards().front().subspan](int v) {
    return subspan(0, 0) + v;
  });
}
} // namespace cuda::experimental
