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

#include <cuda/__iterator/zip_iterator.h>
#include <cuda/std/__algorithm/find_if.h>
#include <cuda/std/__barrier/barrier.h>
#include <cuda/std/__tuple_dir/apply.h>

#include <cuda/experimental/__sharded_mdspan/sharded_mdspan.hpp>

#include <cstddef>
#include <thread>
#include <utility>

namespace cuda::experimental
{
class host_thread_group
{
public:
  host_thread_group(std::uint32_t rank, std::uint32_t size)
      : rank_{rank}
      , size_{size}
  {}

private:
  ::std::uint32_t rank_{};
  ::std::uint32_t size_{};
};

namespace detail
{
template <typename... ShardedMDSpanT>
::cuda::std::tuple<typename ShardedMDSpanT::shard_type*...> find_owned_shards(const ShardedMDSpanT&... sharded_mdspan)
{
  const auto zip_begin = ::cuda::make_zip_iterator(sharded_mdspan.shards().begin()...);
  const auto zip_end   = ::cuda::make_zip_iterator(sharded_mdspan.shards().end()...);
  const auto ret =
    ::cuda::std::find_if(zip_begin, zip_end, [thread_id = ::std::this_thread::get_id()](const auto& tup) {
      return ::cuda::std::apply(
        [&](auto&&... shards) {
          return ((shards.owning_thread_id == thread_id) && ...);
        },
        tup);
    });

  if (ret == zip_end)
  {
    throw ::std::runtime_error{"This thread owns no shards"};
  }

  return ret.__iterators();
}
} // namespace detail

template <typename MDSpan, typename F>
void transform(const host_thread_group& group,
               const basic_sharded_mdspan<MDSpan>& in_mdspan,
               const basic_sharded_mdspan<MDSpan>& out_mdspan,
               F&& functor)
{
  const auto [in_shard_it, out_shard_it] = detail::find_owned_shards(in_mdspan, out_mdspan);

  thrust::transform(in_shard_it->subspan.data(), out_shard_it->subspan.data(), ::std::forward<F>(functor));
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
