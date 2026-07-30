// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include <nccl.h>

namespace mgmn
{
//! Throw on a failed NCCL call, formatting the NCCL error string.
inline void check_nccl(ncclResult_t status, const char* operation)
{
  if (status != ncclSuccess)
  {
    throw std::runtime_error(std::string{operation} + ": " + ncclGetErrorString(status));
  }
}

//! Owning device allocation obtained from `ncclMemAlloc`.
//!
//! Memory that is to be registered as an NCCL symmetric window must come from `ncclMemAlloc`:
//! `ncclCommWindowRegister` resolves the backing allocation via `cuMemGetAddressRange`, which fails
//! with `invalid argument` for stream-ordered memory-pool allocations (`cudaMallocAsync` and
//! anything built on it, including `cuda::make_buffer`). `ncclMemAlloc` returns a plain VA-backed
//! allocation with the properties NCCL's symmetric memory path requires.
template <typename T>
class nccl_buffer
{
public:
  nccl_buffer() = default;

  explicit nccl_buffer(std::size_t count)
      : count_{count}
  {
    void* pointer = nullptr;
    check_nccl(::ncclMemAlloc(&pointer, count * sizeof(T)), "ncclMemAlloc");
    data_ = static_cast<T*>(pointer);
  }

  nccl_buffer(const nccl_buffer&)            = delete;
  nccl_buffer& operator=(const nccl_buffer&) = delete;

  nccl_buffer(nccl_buffer&& other) noexcept
      : data_{std::exchange(other.data_, nullptr)}
      , count_{std::exchange(other.count_, 0)}
  {}

  nccl_buffer& operator=(nccl_buffer&& other) noexcept
  {
    if (this != &other)
    {
      reset();
      data_  = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }

  ~nccl_buffer()
  {
    reset();
  }

  [[nodiscard]] T* data() const noexcept
  {
    return data_;
  }

  [[nodiscard]] std::size_t size() const noexcept
  {
    return count_;
  }

  [[nodiscard]] std::size_t size_bytes() const noexcept
  {
    return count_ * sizeof(T);
  }

private:
  //! Deallocation failures are not actionable during teardown and must not escape the destructor,
  //! so the result is deliberately discarded.
  void reset() noexcept
  {
    if (data_ != nullptr)
    {
      static_cast<void>(::ncclMemFree(data_));
      data_  = nullptr;
      count_ = 0;
    }
  }

  T* data_{};
  std::size_t count_{};
};
} // namespace mgmn
