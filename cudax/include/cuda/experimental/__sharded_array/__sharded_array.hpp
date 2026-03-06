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

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/std/mdspan>
#include <cuda/std/span>
#include <cuda/std/variant>

#include <cuda/experimental/__device/logical_device.cuh>

#include <vector>

namespace cuda::experimental
{
struct host_memory
{};

class memory
{
public:
  memory() = default;

  memory(host_memory)
      : proc_{std::in_place_type<host_memory>}
  {}

  memory(logical_device dev)
      : proc_{std::move(dev)}
  {}

  memory(::cuda::device_ref dev)
      : memory{logical_device{dev}}
  {}

private:
  ::cuda::std::variant<host_memory, logical_device> proc_{};
};

template <typename MDSpan>
class shard
{
public:
  using mdspan_type = MDSpan;

  mdspan_type mdspan{};
  memory proc{};
};

template <typename MDSpan>
explicit shard(const MDSpan&, memory) -> shard<MDSpan>;

template <typename MDSpan>
class basic_sharded_mdarray
{
public:
  using mdspan_type = MDSpan;
  using shard_type  = shard<MDSpan>;

  [[nodiscard]] ::cuda::std::span<const shard_type> shards() const
  {
    return shards_;
  }

  [[nodiscard]] ::cuda::std::span<shard_type> shards()
  {
    return shards_;
  }

private:
  ::std::vector<shard_type> shards_{};
};

template <typename MDSpan, typename... Rest>
basic_sharded_mdarray<MDSpan> make_sharded_mdarray(shard<MDSpan> first, Rest&&... rest)
{
  return basic_sharded_mdarray<MDSpan>{std::move(first), std::move(rest)...};
}

template <typename MDSpan, typename F>
inline void transform(const basic_sharded_mdarray<MDSpan>& mdarray, F&& functor)
{
  for (auto&& shard : mdarray.shards())
  {
  }
}

inline void foo()
{
  auto v1 = thrust::device_vector<int>{4};
  auto v2 = thrust::device_vector<int>{4};

  auto v1_span = ::cuda::std::mdspan{thrust::raw_pointer_cast(v1.data()), 1, 2};
  auto v2_span = ::cuda::std::mdspan{thrust::raw_pointer_cast(v2.data()), 4, 5};

  auto sharded = make_sharded_mdarray(shard{v1_span, logical_device{0}}, shard{v2_span, host_memory{}});

  transform(sharded, [] {});
}
} // namespace cuda::experimental
