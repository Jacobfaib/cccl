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
#include <cuda/std/__algorithm/any_of.h>
#include <cuda/std/__numeric/accumulate.h>
#include <cuda/std/mdspan>
#include <cuda/std/span>
#include <cuda/std/variant>

#include <cuda/experimental/__device/logical_device.cuh>

#include <stdexcept>
#include <vector>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <typename T>
class fancy_accessor;

template <typename Element, typename Tag, typename Reference, typename Derived>
class fancy_accessor<thrust::pointer<Element, Tag, Reference, Derived>>
{
  using fancy_pointer_type = thrust::pointer<Element, Tag, Reference, Derived>;

public:
  using offset_policy    = fancy_accessor;
  using element_type     = typename fancy_pointer_type::value_type;
  using reference        = typename fancy_pointer_type::reference;
  using data_handle_type = fancy_pointer_type;

  constexpr fancy_accessor() noexcept = default;

  template <class U>
  constexpr fancy_accessor(fancy_accessor<thrust::device_ptr<U>>) noexcept
  {}

  [[nodiscard]] constexpr reference access(data_handle_type __p, size_t __i) const noexcept
  {
    return __p[__i];
  }
  [[nodiscard]] constexpr data_handle_type offset(data_handle_type __p, size_t __i) const noexcept
  {
    return __p + __i;
  }
};

_CCCL_TEMPLATE(typename Element, typename Tag, typename Reference, typename Derived, class... _OtherIndexTypes)
_CCCL_REQUIRES((sizeof...(_OtherIndexTypes) > 0) _CCCL_AND(is_convertible_v<_OtherIndexTypes, size_t>&&... && true))
_CCCL_HOST_DEVICE explicit mdspan(thrust::pointer<Element, Tag, Reference, Derived> ptr, _OtherIndexTypes...)
  -> mdspan<Element,
            extents<size_t, __maybe_static_ext<_OtherIndexTypes>...>,
            layout_left,
            fancy_accessor<thrust::pointer<Element, Tag, Reference, Derived>>>;

_CCCL_END_NAMESPACE_CUDA_STD

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

struct host_memory
{
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
  using size_type   = typename mdspan_type::size_type;
  using shard_type  = shard<MDSpan>;

  [[nodiscard]] ::cuda::std::span<const shard_type> shards() const;
  [[nodiscard]] ::cuda::std::span<shard_type> shards();

  [[nodiscard]] size_type size() const;
  [[nodiscard]] bool empty() const;

private:
  ::std::vector<shard_type> shards_{};
};

template <typename M>
::cuda::std::span<const typename basic_sharded_mdarray<M>::shard_type> basic_sharded_mdarray<M>::shards() const
{
  return shards_;
}

template <typename M>
::cuda::std::span<typename basic_sharded_mdarray<M>::shard_type> basic_sharded_mdarray<M>::shards()
{
  return shards_;
}

template <typename M>
typename basic_sharded_mdarray<M>::size_type basic_sharded_mdarray<M>::size() const
{
  size_type ret = 0;

  for (auto&& s : shards())
  {
    ret += s.subspan.size();
  }
  return ret;
}

template <typename M>
bool basic_sharded_mdarray<M>::empty() const
{
  return size() == 0;
}

// ==========================================================================================

template <typename MDSpan, typename... Rest>
basic_sharded_mdarray<MDSpan> make_sharded_mdarray(shard<MDSpan> first, Rest&&... rest)
{
  return basic_sharded_mdarray<MDSpan>{std::move(first), std::move(rest)...};
}

template <typename MDSpan, typename F>
void transform(const basic_sharded_mdarray<MDSpan>& in_mdarray,
               const basic_sharded_mdarray<MDSpan>& out_mdarray,
               F&& functor)
{
  if (in_mdarray.shards().empty())
  {
    return;
  }

  auto&& first_mem = in_mdarray.shards().front().proc;

  auto zipper = make_zip_iterator(in_mdarray.shards(), out_mdarray.shards());

  if (::cuda::std::any_of(zipper.begin(), zipper.end(), [&](const auto& tup) {
        auto&& [in_arr, out_arr] = tup;

        return in_arr.proc != first_mem || out_arr.proc != first_mem;
      }))
  {
    throw ::std::runtime_error{"memories don't match"};
  }

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

  auto v1_span = ::cuda::std::mdspan{v1.data(), 1, 2};
  auto v2_span = ::cuda::device_mdspan{thrust::raw_pointer_cast(v2.data()), 4, 5};

  auto sharded = make_sharded_mdarray(shard{v1_span, logical_device{0}}, shard{v2_span, host_memory{}});

  transform(sharded, sharded, [] {});
}
} // namespace cuda::experimental
