//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___THREAD_GROUP_THREAD_GROUP_H
#define _CUDA_EXPERIMENTAL___THREAD_GROUP_THREAD_GROUP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/all_devices.h>
#include <cuda/std/__barrier/barrier.h>
#include <cuda/std/__cstddef/byte.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__memory/align.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/atomic>
#include <cuda/std/span>

#include <cuda/experimental/__device/logical_device.cuh>

#include <memory>
#include <stdexcept>
#include <vector>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
class thread_group
{
public:
  using size_type = ::cuda::std::size_t;
  using rank_type = ::cuda::std::size_t;

private:
  static constexpr ::cuda::std::byte __MAGIC_INIT_VALUE{222};
  static constexpr ::cuda::std::byte __MAGIC_CONSTRUCTED_VALUE{0};

  class __shared_data
  {
  public:
    __shared_data() = delete;

    _CCCL_HOST_API explicit __shared_data(
      const ::cuda::std::byte* __raw_ptr, size_type __group_size, ::cuda::std::span<::cuda::std::byte> __mem)
        // Do not initialize __constructed_flag in initializer list, it must be initialized
        // atomically
        : __barrier{static_cast<::cuda::std::ptrdiff_t>(__group_size)}
        , __scratch_mem{__mem}
    {
      // We need the __constructed_flag member to sit at exactly the first byte of the shared
      // memory buffer (not __mem, the original buffer), so that the other threads can be
      // properly notified that we have constructed ourselves. If that is ever not the case,
      // then we will write to the wrong memory location and never release the waiting threads.
      _CCCL_ASSERT(&__constructed_flag == __raw_ptr,
                   "__shared_data constructed at an offset inside shared memory scratch buffer");

      auto __ref = ::cuda::std::atomic_ref<::cuda::std::byte>{__constructed_flag};

      __ref = __MAGIC_CONSTRUCTED_VALUE;
      __ref.notify_all();
    }

    ::cuda::std::byte __constructed_flag;
    ::cuda::std::barrier<> __barrier;
    ::cuda::std::span<::cuda::std::byte> __scratch_mem;
  };

public:
  _CCCL_HOST_API thread_group(rank_type __rank,
                              size_type __size,
                              const ::cuda::__all_devices& __all_dev,
                              ::std::unique_ptr<::cuda::std::byte[]> __shared_mem)
      : thread_group{__rank, __size, {__all_dev.begin(), __all_dev.end()}, ::cuda::std::move(__shared_mem)}
  {}

  _CCCL_HOST_API thread_group(rank_type __rank,
                              size_type __size,
                              ::std::vector<logical_device> __devices,
                              ::std::unique_ptr<::cuda::std::byte[]> __shared_mem)
      : __rank_{__rank}
      , __size_{__size}
      , __devices_{::cuda::std::move(__devices)}
      , __shared_mem_{::cuda::std::move(__shared_mem)}
  {
    // Technically only thread 0 needs to perform any of the pointer aligning, but we let all
    // threads do it so we can error-check the arguments properly.
    auto* const __raw_ptr         = __shared_mem_.get();
    const auto __min_scratch_size = required_scratch_mem_size_(size());
    auto* __scratch_ptr           = static_cast<void*>(__raw_ptr + sizeof(__shared_data));
    auto __capacity               = required_shared_memory_size(size()) - sizeof(__shared_data);

    if (!::cuda::std::align(required_scratch_mem_alignment_(), __min_scratch_size, __scratch_ptr, __capacity))
    {
      _CCCL_THROW(::std::length_error,
                  "Failed to align scratch memory pointer. Likely allocated shared memory is insufficient.");
    }

    // Construct last, to preserve strong exception guarantee.
    if (rank() == 0)
    {
      auto* const __ptr = reinterpret_cast<__shared_data*>(__raw_ptr);
      auto __span = ::cuda::std::span<::cuda::std::byte>{static_cast<::cuda::std::byte*>(__scratch_ptr), __capacity};

      ::cuda::std::__construct_at(__ptr, __raw_ptr, size(), __span);
    }
    else
    {
      // Cannot use barrier because the barrier may not yet be constucteed. Instead, we use the
      // first byte of the shared memory to
      auto __ref = ::cuda::std::atomic_ref<::cuda::std::byte>{__shared_mem[0]};

      __ref.wait(__MAGIC_INIT_VALUE);
    }
  }

  _CCCL_HIDE_FROM_ABI thread_group(const thread_group&)            = delete;
  _CCCL_HIDE_FROM_ABI thread_group& operator=(const thread_group&) = delete;

  _CCCL_HIDE_FROM_ABI thread_group(thread_group&&) noexcept            = default;
  _CCCL_HIDE_FROM_ABI thread_group& operator=(thread_group&&) noexcept = default;

  _CCCL_HOST_API ~thread_group()
  {
#ifndef NDEBUG
    barrier();
#endif
    if (rank() == 0)
    {
      ::cuda::std::destroy_at(&shared_data_());
    }
  }

  [[nodiscard]] _CCCL_HOST_API constexpr rank_type rank() const noexcept
  {
    return __rank_;
  }

  [[nodiscard]] _CCCL_HOST_API constexpr size_type size() const noexcept
  {
    return __size_;
  }

  _CCCL_HOST_API void barrier() const
  {
    if (size() > 1)
    {
      mut_shared_data_().__barrier.arrive_and_wait();
    }
  }

  [[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::span<const logical_device> devices() const noexcept
  {
    return __devices_;
  }

  [[nodiscard]] _CCCL_API static constexpr ::cuda::std::size_t
  required_shared_memory_size(size_type __group_size) noexcept
  {
    return sizeof(__shared_data) + alignof(__shared_data) - 1 + required_scratch_mem_size_(__group_size);
  }

  _CCCL_API static constexpr void* initialize_shared_memory(void* __mem) noexcept
  {
    // If we are in a consteval context then a simple static_cast suffices, but otherwise we
    // must atomically initialize. It is possible that the user may be using one of the
    // participating threads to initialize the memory.
    _CCCL_IF_CONSTEVAL
    {
      *static_cast<::cuda::std::byte*>(__mem) = __MAGIC_INIT_VALUE;
    }
    else
    {
      ::cuda::std::atomic_ref{*static_cast<::cuda::std::byte*>(__mem)} = __MAGIC_INIT_VALUE;
    }
    return __mem;
  }

private:
  [[nodiscard]] _CCCL_API static constexpr ::cuda::std::size_t
  required_scratch_mem_size_(size_type __group_size) noexcept
  {
    // All members of the group have a void*'s worth of private memory
    return __group_size * sizeof(void*);
  }

  [[nodiscard]] _CCCL_API static constexpr ::cuda::std::size_t required_scratch_mem_alignment_() noexcept
  {
    return alignof(::cuda::std::max_align_t);
  }

  [[nodiscard]] _CCCL_HOST_API __shared_data& shared_data_() noexcept
  {
    return reinterpret_cast<__shared_data&>(*__shared_mem_.get());
  }

  [[nodiscard]] _CCCL_HOST_API const __shared_data& shared_data_() const noexcept
  {
    return reinterpret_cast<const __shared_data&>(*__shared_mem_.get());
  }

  [[nodiscard]] _CCCL_HOST_API __shared_data& mut_shared_data_() const noexcept
  {
    return const_cast<thread_group&>(*this).shared_data_();
  }

  rank_type __rank_{};
  size_type __size_{};
  ::std::vector<logical_device> __devices_{};
  ::std::unique_ptr<::cuda::std::byte[]> __shared_mem_{};
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___THREAD_GROUP_THREAD_GROUP_H
