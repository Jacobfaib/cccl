//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_THREAD_GROUP_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_THREAD_GROUP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/all_devices.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__barrier/barrier.h>
#include <cuda/std/__cstddef/byte.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__host_stdlib/memory>
#include <cuda/std/__host_stdlib/stdexcept>
#include <cuda/std/__memory/align.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/atomic>
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
class thread_group
{
  static constexpr ::cuda::std::byte __MAGIC_INIT_VALUE{222};
  static constexpr ::cuda::std::byte __MAGIC_CONSTRUCTED_VALUE{0};

  // Otherwise the construction signaling doesn't work
  static_assert(__MAGIC_INIT_VALUE != __MAGIC_CONSTRUCTED_VALUE);

  class __shared_data
  {
    ::cuda::std::byte __constructed_flag;
    ::cuda::std::atomic<::cuda::std::uint32_t> __destroy_count{};
    ::cuda::std::barrier<> __bar;
    ::cuda::std::span<::cuda::std::byte> __scratch_mem{};

  public:
    _CCCL_HOST_API explicit __shared_data(const ::cuda::std::byte* __raw_ptr,
                                          ::cuda::std::ptrdiff_t __group_size,
                                          ::cuda::std::span<::cuda::std::byte> __mem)
        // Do not initialize __constructed_flag in initializer list, it must be initialized
        // atomically
        : __destroy_count{static_cast<::cuda::std::uint32_t>(__group_size)}
        , __bar{__group_size}
        , __scratch_mem{__mem}
    {
      // We need the __constructed_flag member to sit at exactly the first byte of the shared
      // memory buffer (not __mem, the original buffer), so that the other threads can be
      // properly notified that we have constructed ourselves. This is also why we use a raw
      // byte, and not a atomic<byte>, because we cannot guarantee where the atomic
      // implementation will place the value inside the struct.
      //
      // If this value is ever not at the very first byte, then we will write to the wrong
      // memory location and never release the waiting threads.
      static_assert(offsetof(__shared_data, __constructed_flag) == 0);

      _CCCL_VERIFY(&__constructed_flag == __raw_ptr,
                   "__shared_data constructed at an offset inside shared memory scratch buffer");

      auto __ref = ::cuda::std::atomic_ref{__constructed_flag};

      __ref.store(__MAGIC_CONSTRUCTED_VALUE, ::cuda::std::memory_order_release);
      __ref.notify_all();
    }

    _CCCL_HOST_API bool __can_destroy() noexcept
    {
      return __destroy_count.fetch_sub(1, ::cuda::std::memory_order_release) == 1;
    }

    [[nodiscard]] _CCCL_HOST_API ::cuda::std::barrier<>& __barrier() noexcept
    {
      return __bar;
    }

    [[nodiscard]] _CCCL_HOST_API const ::cuda::std::barrier<>& __barrier() const noexcept
    {
      return __bar;
    }

    template <typename T>
    [[nodiscard]] _CCCL_HOST_API ::cuda::std::span<T> __scratch_mem_as() noexcept
    {
      auto* const ptr = reinterpret_cast<T*>(__scratch_mem.data());

      return {ptr, __scratch_mem.size() / sizeof(T)};
    }

    template <typename T>
    [[nodiscard]] _CCCL_HOST_API ::cuda::std::span<T> __scratch_mem_as() const noexcept
    {
      auto* const ptr = reinterpret_cast<T*>(__scratch_mem.data());

      return {ptr, __scratch_mem.size() / sizeof(T)};
    }

    template <>
    [[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::span<::cuda::std::byte> __scratch_mem_as() noexcept
    {
      return __scratch_mem;
    }

    template <>
    [[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::span<const ::cuda::std::byte> __scratch_mem_as() const noexcept
    {
      return __scratch_mem;
    }
  };

public:
  _CCCL_HOST_API thread_group(const ::cuda::__all_devices& __all_devices)
      : thread_group{/*__rank*/ 0,
                     /*__group_size*/ 1,
                     static_cast<::cuda::std::uint32_t>(__all_devices.size()),
                     __singleton_shared_mem()}
  {}

  _CCCL_HOST_API thread_group(::cuda::std::uint32_t __rank,
                              ::cuda::std::uint32_t __group_size,
                              ::cuda::std::uint32_t __num_gpus,
                              ::std::shared_ptr<::cuda::std::byte[]> __shared_mem)
      : __rank_{__rank}
      , __group_size_{__group_size}
      , __shared_mem_{__construct_shared_data(__rank, __group_size, ::cuda::std::move(__shared_mem))}
  {
    _CCCL_TRY
    {
      const auto __u32_scratch = shared_data_()->__scratch_mem_as<::cuda::std::uint32_t>();
      const auto __ref         = ::cuda::std::atomic_ref{__u32_scratch[rank()]};

      __ref.store(__num_gpus, ::cuda::std::memory_order_relaxed);

      barrier();

      for (::cuda::std::uint32_t __i = 0; __i < size(); ++__i)
      {
        const auto __s = __u32_scratch[__i];

        if (__i < rank())
        {
          __sub_idx_begin_ += __s;
        }
        __stride_ += __s;
      }
      __sub_idx_end_ = __sub_idx_begin_ + __num_gpus;
    }
    _CCCL_CATCH_ALL
    {
      // Strong exception guarantee. There is an argument to be made here to just call
      // cuda::std::terminate(), because if any of the threads throw, then the program *will*
      // eventually deadlock on a barrier (possibly even the one above).
      __reset();
      _CCCL_RETHROW;
    }
  }

  // This object may not be copied because we have no way of reliably keeping the destroy
  // refcount accurate.
  _CCCL_HIDE_FROM_ABI thread_group& operator=(const thread_group&) = delete;
  _CCCL_HIDE_FROM_ABI thread_group(const thread_group&)            = delete;

  _CCCL_HOST_API thread_group& operator=(thread_group&& __other) noexcept
  {
    if (this == &__other)
    {
      return *this;
    }

    // Must __reset here to ensure the __shared_data structure is properly destroyed.
    __reset();

    __rank_          = ::cuda::std::exchange(__other.__rank_, {});
    __group_size_    = ::cuda::std::exchange(__other.__group_size_, {});
    __sub_idx_begin_ = ::cuda::std::exchange(__other.__sub_idx_begin_, {});
    __sub_idx_end_   = ::cuda::std::exchange(__other.__sub_idx_end_, {});
    __stride_        = ::cuda::std::exchange(__other.__stride_, {});
    __shared_mem_    = ::cuda::std::move(__other.__shared_mem_);
    return *this;
  }

  // We can = default the move constructor because we have nothing to __reset()
  _CCCL_HIDE_FROM_ABI thread_group(thread_group&&) noexcept = default;

  _CCCL_HOST_API ~thread_group()
  {
    __reset();
  }

  [[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::uint32_t rank() const noexcept
  {
    return __rank_;
  }

  [[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::uint32_t size() const noexcept
  {
    return __group_size_;
  }

  [[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::uint32_t index_begin() const noexcept
  {
    return __sub_idx_begin_;
  }

  [[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::uint32_t index_end() const noexcept
  {
    return __sub_idx_end_;
  }

  [[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::uint32_t stride() const noexcept
  {
    return __stride_;
  }

  template <typename __F>
  _CCCL_HOST_API void dispatch(::cuda::std::uint32_t __size, __F&& fn) const
  {
    for (::cuda::std::uint32_t __off = 0; __off < __size; __off += stride())
    {
      for (auto __i = index_begin() + __off; __i < ::cuda::std::min(__size, index_end() + __off); ++__i)
      {
        ::cuda::std::forward<__F>(fn)(__i);
      }
    }
  }

  _CCCL_HOST_API void barrier() const
  {
    if (size() > 1)
    {
      mut_shared_data_()->__barrier().arrive_and_wait();
    }
  }

  [[nodiscard]] _CCCL_HOST_API static constexpr ::cuda::std::size_t
  required_shared_memory_size(::cuda::std::uint32_t __group_size) noexcept
  {
    return sizeof(__shared_data) + alignof(__shared_data) - 1 + __required_scratch_mem_size(__group_size);
  }

  _CCCL_HOST_API static constexpr void* initialize_shared_memory(void* __mem) noexcept
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
      ::cuda::std::atomic_ref{*static_cast<::cuda::std::byte*>(__mem)}.store(
        __MAGIC_INIT_VALUE, ::cuda::std::memory_order_release);
    }
    return __mem;
  }

private:
  [[nodiscard]] _CCCL_HOST_API static constexpr ::cuda::std::size_t
  __required_scratch_mem_size(::cuda::std::uint32_t __group_size) noexcept
  {
    // All members of the group have a uint64_t's worth of private memory
    return __group_size * sizeof(::cuda::std::uint64_t);
  }

  [[nodiscard]] _CCCL_HOST_API static constexpr ::cuda::std::uint32_t __required_scratch_mem_alignment() noexcept
  {
    return alignof(::cuda::std::max_align_t);
  }

  [[nodiscard]] _CCCL_HOST_API __shared_data* shared_data_() noexcept
  {
    return reinterpret_cast<__shared_data*>(__shared_mem_.get());
  }

  [[nodiscard]] _CCCL_HOST_API const __shared_data* shared_data_() const noexcept
  {
    return reinterpret_cast<const __shared_data*>(__shared_mem_.get());
  }

  [[nodiscard]] _CCCL_HOST_API __shared_data* mut_shared_data_() const noexcept
  {
    return const_cast<thread_group&>(*this).shared_data_();
  }

  _CCCL_HOST_API void __reset() noexcept
  {
    // We could have been moved-from, so need to test __sd
    if (auto* const __sd = shared_data_())
    {
      // There is a small micro-optimization we can employ here. Normally, reference count
      // decrements need to be memory_order_acq_rel or memory_order_seq_cst because the last
      // thread to decrement also needs to see all the write effects of prior threads. We can,
      // however, relax the decrements to memory_order_release, so long as we acquire before
      // making any changes.
      if (__sd->__can_destroy())
      {
        // Hence this threadfence.
        ::cuda::std::atomic_thread_fence(::cuda::std::memory_order_acquire);
        ::cuda::std::destroy_at(__sd);
      }
      __shared_mem_.reset();
    }
    __rank_          = 0;
    __group_size_    = 0;
    __sub_idx_begin_ = 0;
    __sub_idx_end_   = 0;
    __stride_        = 0;
  }

  [[nodiscard]] _CCCL_HOST_API static ::std::shared_ptr<::cuda::std::byte[]> __construct_shared_data(
    ::cuda::std::uint32_t __rank, ::cuda::std::uint32_t __size, ::std::shared_ptr<::cuda::std::byte[]> __shared_mem)
  {
    // Technically only thread 0 needs to perform any of the pointer aligning, but we let all
    // threads do it so we can error-check the arguments properly.
    auto* const __raw_ptr         = __shared_mem.get();
    const auto __min_scratch_size = __required_scratch_mem_size(__size);
    auto* __scratch_ptr           = static_cast<void*>(__raw_ptr + sizeof(__shared_data));
    auto __capacity               = required_shared_memory_size(__size) - sizeof(__shared_data);

    if (!::cuda::std::align(__required_scratch_mem_alignment(), __min_scratch_size, __scratch_ptr, __capacity))
    {
      _CCCL_THROW(::std::length_error,
                  "Failed to align scratch memory pointer. Likely allocated shared memory is insufficient.");
    }

    // Construct last, to preserve strong exception guarantee.
    if (__rank == 0)
    {
      auto* const __ptr = reinterpret_cast<__shared_data*>(__raw_ptr);
      auto __span = ::cuda::std::span<::cuda::std::byte>{static_cast<::cuda::std::byte*>(__scratch_ptr), __capacity};

      ::cuda::std::__construct_at(__ptr, __raw_ptr, static_cast<::cuda::std::ptrdiff_t>(__size), __span);
    }
    else
    {
      // Cannot use barrier because the barrier may not yet be constucteed. Instead, we use the
      // first byte of the shared memory to
      auto __ref = ::cuda::std::atomic_ref{__shared_mem[0]};

      __ref.wait(__MAGIC_INIT_VALUE, ::cuda::std::memory_order_acquire);
    }
    return __shared_mem;
  }

  [[nodiscard]] _CCCL_HOST_API static const ::std::shared_ptr<::cuda::std::byte[]>& __singleton_shared_mem()
  {
    static const auto __mem = [] {
      auto __ret =
        ::std::shared_ptr<::cuda::std::byte[]>{new ::cuda::std::byte[required_shared_memory_size(/*group_size*/ 1)]{}};

      initialize_shared_memory(__ret.get());
      return __ret;
    }();

    return __mem;
  }

  ::cuda::std::uint32_t __rank_{};
  ::cuda::std::uint32_t __group_size_{};
  ::cuda::std::uint32_t __sub_idx_begin_{};
  ::cuda::std::uint32_t __sub_idx_end_{};
  ::cuda::std::uint32_t __stride_{};
  ::std::shared_ptr<::cuda::std::byte[]> __shared_mem_{};
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___THREAD_GROUP_THREAD_GROUP_H
