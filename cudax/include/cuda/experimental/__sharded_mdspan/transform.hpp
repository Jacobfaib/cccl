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

#include <cuda/experimental/__sharded_mdspan/sharded_mdspan.hpp>

namespace cuda::experimental
{
template <typename MDSpan, typename F>
void transform(const basic_sharded_mdspan<MDSpan>& in_mdarray,
               const basic_sharded_mdspan<MDSpan>& out_mdarray,
               F&& functor)
{
  auto zipper = make_zip_iterator(in_mdarray.shards(), out_mdarray.shards());

  for (auto&& [in_shard, out_shard] : zipper)
  {
    const auto _ = in_shard.proc.activate_guard();

    thrust::transform(in_shard.subspan, out_shard.subspan, std::forward<F>(functor));
  }
}

inline void foo()
{
  auto v1 = thrust::device_vector<int>{4};
  auto v2 = thrust::device_vector<int>{4};

  auto v1_span = ::cuda::std::mdspan{thrust::raw_pointer_cast(v1.data()), 1, 2};
  auto v2_span = ::cuda::device_mdspan{thrust::raw_pointer_cast(v2.data()), 4, 5};

  auto sharded = basic_sharded_mdarray{{v1_span, logical_device{0}}, {v2_span, host_memory{}}};

  transform(sharded, sharded, [] {});
}
} // namespace cuda::experimental
