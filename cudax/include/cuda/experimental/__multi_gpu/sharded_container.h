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

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__functional/call_or.h>
#include <cuda/__memory_pool/device_memory_pool.h>
#include <cuda/__memory_resource/any_resource.h>
#include <cuda/__memory_resource/get_memory_resource.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/span>

#include <cuda/experimental/__device/logical_device.cuh>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <typename Iterator>
class shard
{
public:
  using iterator_type     = Iterator;
  using resource_type     = ::cuda::mr::any_resource<::cuda::mr::device_accessible>;
  using resource_ref_type = ::cuda::mr::resource_ref<::cuda::mr::device_accessible>;

  _CCCL_TEMPLATE(typename _Range)
  _CCCL_REQUIRES(::cuda::std::ranges::range<_Range>)
  shard(_Range&& __range, logical_device __dev, resource_type __res)
      : shard{::cuda::std::ranges::begin(__range), ::cuda::std::ranges::end(__range), __dev, ::cuda::std::move(__res)}
  {}

  _CCCL_TEMPLATE(typename _Range)
  _CCCL_REQUIRES(::cuda::std::ranges::range<_Range>)
  shard(_Range&& __range, device_ref __dev, resource_type __res)
      : shard{::cuda::std::forward<_Range>(__range), logical_device{__dev}, ::cuda::std::move(__res)}
  {}

  _CCCL_TEMPLATE(typename _Range)
  _CCCL_REQUIRES(::cuda::std::ranges::range<_Range>)
  shard(_Range&& __range, logical_device __dev)
      : shard{
          ::cuda::std::ranges::begin(__range),
          ::cuda::std::ranges::end(__range),
          ::cuda::std::move(__dev),
          ::cuda::__call_or(
            ::cuda::mr::get_memory_resource, ::cuda::device_default_memory_pool(__dev.underlying_device()), __range)}
  {}

  _CCCL_TEMPLATE(typename _Range)
  _CCCL_REQUIRES(::cuda::std::ranges::range<_Range>)
  shard(_Range&& __range, device_ref __dev)
      : shard{::cuda::std::forward<_Range>(__range), logical_device{__dev}}
  {}

  shard(iterator_type __b, iterator_type __e, logical_device __dev, resource_type __res)
      : __begin_{__b}
      , __end_{__e}
      , __device_{__dev}
      , __resource_{::cuda::std::move(__res)}
  {}

  // To disallow range-for interface to the shard, since this would be surprising
  [[nodiscard]] constexpr iterator_type iter_begin() const
  {
    return __begin_;
  }

  [[nodiscard]] constexpr iterator_type iter_end() const
  {
    return __begin_;
  }

  [[nodiscard]] constexpr const logical_device& device() const
  {
    return __begin_;
  }

  [[nodiscard]] constexpr resource_ref_type resource() const
  {
    return __resource_;
  }

private:
  iterator_type __begin_;
  iterator_type __end_;
  logical_device __device_;
  resource_type __resource_;
};

template <typename Range>
_CCCL_HOST_API shard(Range&&, logical_device, ::cuda::mr::any_resource<::cuda::mr::device_accessible>)
  -> shard<::cuda::std::ranges::iterator_t<Range>>;

template <typename Range>
_CCCL_HOST_API shard(Range&&, logical_device) -> shard<::cuda::std::ranges::iterator_t<Range>>;

template <typename Range>
_CCCL_HOST_API shard(Range&&, device_ref) -> shard<::cuda::std::ranges::iterator_t<Range>>;

template <typename _Iterator, ::cuda::std::size_t _N = ::cuda::std::dynamic_extent>
class sharded_view
{
public:
  using shard_type = shard<_Iterator>;
  using size_type  = ::cuda::std::size_t;

  constexpr sharded_view() = default;

  constexpr explicit sharded_view(::cuda::std::span<shard_type, _N> __shards) noexcept
      : __shards_{__shards}
  {}

  [[nodiscard]] ::cuda::std::span<const shard_type> shards() const noexcept
  {
    return __shards_;
  }

  [[nodiscard]] ::cuda::std::span<shard_type> shards() noexcept
  {
    return __shards_;
  }

  [[nodiscard]] size_type size() const noexcept
  {
    size_type __ret = 0;

    for (auto&& __s : shards())
    {
      __ret += ::cuda::std::distance(__s.iter_begin(), __s.iter_end());
    }
    return __ret;
  }

  [[nodiscard]] bool empty() const noexcept
  {
    return size() == 0;
  }

private:
  ::cuda::std::span<shard_type, _N> __shards_{};
};

template <typename _Iterator>
class sharded_buffer;

template <typename _Tp>
inline constexpr bool __is_sharded_buf = false;

template <typename _Tp>
inline constexpr bool __is_sharded_buf<sharded_buffer<_Tp>> = true;

template <typename _Iterator>
class sharded_buffer : public sharded_view<_Iterator>
{
  using __base = sharded_view<_Iterator>;

public:
  using typename __base::shard_type;

  constexpr sharded_buffer() = default;

  constexpr explicit sharded_buffer(::std::vector<shard_type> __shards) noexcept
      : __base{__shards}
      , __owned_shards_{::cuda::std::move(__shards)}
  {}

  _CCCL_TEMPLATE(typename _FirstShard, typename... _Shards)
  _CCCL_REQUIRES((!__is_sharded_buf<_FirstShard>) _CCCL_AND ::cuda::std::is_constructible_v<shard_type, _FirstShard>
                   _CCCL_AND((::cuda::std::is_constructible_v<shard_type, _Shards> && ...)))
  constexpr explicit sharded_buffer(_FirstShard&& __first, _Shards&&... __shards)
      : sharded_buffer{::std::vector<shard_type>{
          ::cuda::std::forward<_FirstShard>(__first), ::cuda::std::forward<_Shards>(__shards)...}}
  {}

private:
  ::std::vector<shard_type> __owned_shards_{};
};

_CCCL_TEMPLATE(typename _FirstShard, typename... _Shards)
_CCCL_REQUIRES(!__is_sharded_buf<_FirstShard>)
_CCCL_HOST_API sharded_buffer(_FirstShard, _Shards...) -> sharded_buffer<typename _FirstShard::iterator_type>;
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
