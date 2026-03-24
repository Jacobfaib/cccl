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

#include <cuda/__driver/driver_api.h>
#include <cuda/__memory_resource/any_resource.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/span>
#include <cuda/std/variant>

#include <cuda/experimental/__device/logical_device.cuh>

#include <vector>

namespace cuda::experimental
{
class host_memory
{
public:
  using resource_type     = ::cuda::mr::any_resource<::cuda::mr::host_accessible>;
  using resource_ref_type = ::cuda::mr::resource_ref<::cuda::mr::host_accessible>;

  [[nodiscard]] resource_ref_type memory_resource() const
  {
    return resource_ref_type{resource_};
  }

  friend bool operator==(const host_memory& lhs, const host_memory& rhs) noexcept
  {
    return lhs.resource_ == rhs.resource_;
  }

  friend bool operator!=(const host_memory& lhs, const host_memory& rhs) noexcept
  {
    return !(lhs == rhs);
  }

private:
  resource_type resource_{};
};

class device_memory
{
public:
  explicit device_memory(logical_device device)
      : device_{::cuda::std::move(device)}
  {}

  explicit device_memory(::cuda::device_ref ref)
      : device_memory{logical_device{ref}}
  {}

  [[nodiscard]] const logical_device& device() const
  {
    return device_;
  }

  [[nodiscard]] ::cuda::mr::resource_ref<::cuda::mr::device_accessible> resource() const
  {
    return resource_;
  }

  friend bool operator==(const device_memory& lhs, const device_memory& rhs) noexcept
  {
    return lhs.device() == rhs.device() && lhs.resource() == rhs.resource();
  }

  friend bool operator!=(const device_memory& lhs, const device_memory& rhs) noexcept
  {
    return !(lhs == rhs);
  }

private:
  logical_device device_;
  ::cuda::mr::any_resource<::cuda::mr::device_accessible> resource_{};
};

class memory
{
public:
  memory() = default;

  memory(host_memory mem)
      : memory_{::cuda::std::move(mem)}
  {}

  memory(device_memory dev)
      : memory_{::cuda::std::move(dev)}
  {}

  memory(::cuda::device_ref dev)
      : memory{device_memory{dev}}
  {}

  [[nodiscard]] bool operator==(const memory& other) const noexcept
  {
    return other.memory_ == memory_;
  }

  [[nodiscard]] bool operator!=(const memory& other) const noexcept
  {
    return (*this == other);
  }

private:
  ::cuda::std::variant<host_memory, device_memory> memory_{};
};

template <typename MDSpan>
class basic_sharded_mdspan
{
public:
  using mdspan_type = MDSpan;
  using value_type  = typename mdspan_type::value_type;
  using size_type   = typename mdspan_type::size_type;

  class shard_type
  {
  public:
    mdspan_type mdspan{};
    memory proc{};
  };

  [[nodiscard]] ::cuda::std::span<const shard_type> shards() const;
  [[nodiscard]] ::cuda::std::span<shard_type> shards();

  [[nodiscard]] size_type size() const;
  [[nodiscard]] bool empty() const;

private:
  ::std::vector<shard_type> shards_{};
};

template <typename Shard, typename... Rest>
basic_sharded_mdspan(Shard, Rest...) -> basic_sharded_mdspan<typename Shard::mdspan_type>;

template <typename M>
::cuda::std::span<const typename basic_sharded_mdspan<M>::shard_type> basic_sharded_mdspan<M>::shards() const
{
  return shards_;
}

template <typename M>
::cuda::std::span<typename basic_sharded_mdspan<M>::shard_type> basic_sharded_mdspan<M>::shards()
{
  return shards_;
}

template <typename M>
typename basic_sharded_mdspan<M>::size_type basic_sharded_mdspan<M>::size() const
{
  size_type ret = 0;

  for (auto&& s : shards())
  {
    ret += s.mdspan.size();
  }
  return ret;
}

template <typename M>
bool basic_sharded_mdspan<M>::empty() const
{
  return size() == 0;
}

// ==========================================================================================

template <typename MDSpan, typename... Rest>
basic_sharded_mdspan<MDSpan> make_sharded_mdspan(typename basic_sharded_mdspan<MDSpan>::shard first, Rest&&... rest)
{
  return basic_sharded_mdspan<MDSpan>{::cuda::std::move(first), ::cuda::std::forward<Rest>(rest)...};
}
} // namespace cuda::experimental
