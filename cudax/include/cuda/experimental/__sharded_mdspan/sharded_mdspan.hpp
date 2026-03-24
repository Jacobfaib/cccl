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

#include <cuda/__driver/driver_api.h>
#include <cuda/__iterator/zip_iterator.h>
#include <cuda/devices>
#include <cuda/mdspan>
#include <cuda/std/span>
#include <cuda/std/variant>

#include <cuda/experimental/__device/logical_device.cuh>

#include <stdexcept>
#include <vector>

namespace cuda::experimental
{
namespace detail
{
template <typename... T>
class overload : public T...
{
public:
  using T::operator()...;
};

template <typename... T>
overload(T...) -> overload<T...>;
} // namespace detail

class host_memory
{
public:
  constexpr bool operator==(const host_memory&) noexcept
  {
    return true;
  }

  constexpr bool operator!=(const host_memory&) noexcept
  {
    return false;
  }
};

class memory
{
  class activated_memory
  {
  public:
    activated_memory() = delete;

    explicit activated_memory(memory& mem)
        : mem_{&mem}
    {
      mem_->activate_();
    }

    ~activated_memory()
    {
      mem_->deactivate_();
    }

    activated_memory(activated_memory&)            = delete;
    activated_memory& operator=(activated_memory&) = delete;

  private:
    memory* mem_{};
  };

  friend activated_memory;

public:
  memory() = default;

  memory(host_memory)
      : proc_{::std::in_place_type<host_memory>}
  {}

  memory(logical_device dev)
      : proc_{::std::move(dev)}
  {}

  memory(::cuda::device_ref dev)
      : memory{logical_device{dev}}
  {}

  [[nodiscard]] activated_memory activate_guard()
  {
    return activated_memory{*this};
  }

  [[nodiscard]] bool operator==(const memory& other) const noexcept
  {
    return other.proc_ == proc_;
  }

  [[nodiscard]] bool operator!=(const memory& other) const noexcept
  {
    return (*this == other);
  }

private:
  void activate_()
  {
    ::cuda::std::visit(proc_,
                       detail::overload{[](const host_memory&) {},
                                        [](const logical_device& dev) {
                                          ::cuda::__driver::__ctxPush(dev.context());
                                        }});
  }

  void deactivate_()
  {
    ::cuda::std::visit(proc_,
                       detail::overload{[](const host_memory&) {},
                                        [](const logical_device&) {
                                          ::cuda::__driver::__ctxPop();
                                        }});
  }

  ::cuda::std::variant<host_memory, logical_device> proc_{};
};

template <typename MDSpan>
class basic_sharded_mdspan
{
public:
  using mdspan_type = MDSpan;
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
    ret += s.subspan.size();
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
  return basic_sharded_mdspan<MDSpan>{std::move(first), std::move(rest)...};
}
} // namespace cuda::experimental
