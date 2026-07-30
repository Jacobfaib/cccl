// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

//! A `cuda::mr::resource` that allocates inside a single GPU locality domain.
//!
//! Backed by a memory pool created with `CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN`, so every
//! allocation's backing store is placed in the requested domain. Pairing this with a green context
//! built from the same domain id (see `locality_domain.hpp`) keeps a rank's data in the memory that
//! its SMs are closest to.
//!
//! Modelling it as a memory resource rather than a bare allocation helper lets it be handed
//! directly to `cuda::make_buffer`, which is how the benchmark gets per-domain `cuda::device_buffer`s.
//!
//! Split into a reference and an owner, mirroring `cuda::device_memory_pool_ref` /
//! `cuda::device_memory_pool`. `cuda::buffer` stores a *copy* of the resource it was given, so the
//! type handed to `make_buffer` must be copyable; a non-owning ref is trivially so, while the owner
//! keeps sole responsibility for destroying the pool.
//!
//! The allocation machinery is inherited rather than reimplemented: once the localized pool exists
//! it is an ordinary `cudaMemPool_t`, so `cuda::__memory_pool_base` already supplies the
//! stream-ordered and synchronous allocate/deallocate pairs and pool equality, and
//! `cuda::mr::memory_resource_base` supplies the resource-concept glue. All that is left here is
//! creating the pool with the right location and carrying the domain identity.

#include <cuda/__memory_pool/device_memory_pool.h>
#include <cuda/memory_resource>

#include <cuda.h>

#include "locality_domain.hpp"

namespace mgmn
{
//! Non-owning view of a memory pool localized to one locality domain.
//!
//! Copyable, which is what lets it be handed to `cuda::make_buffer`. The caller must keep the
//! owning `locality_domain_resource` alive for as long as any buffer drawn from it.
class locality_domain_resource_ref
    : public ::cuda::__memory_pool_base
    , public ::cuda::mr::memory_resource_base<locality_domain_resource_ref>
{
public:
  // Not `constexpr`: the `__memory_pool_base` base class constructor is not.
  explicit locality_domain_resource_ref(::cudaMemPool_t pool, cuda::device_ref device, unsigned int domain) noexcept
      : ::cuda::__memory_pool_base{pool}
      , device_{device}
      , domain_{domain}
  {}

  [[nodiscard]] constexpr cuda::device_ref device() const noexcept
  {
    return device_;
  }

  //! The locality domain this resource allocates in.
  [[nodiscard]] constexpr unsigned int domain() const noexcept
  {
    return domain_;
  }

  //! Enables the `device_accessible` property, which is what lets `make_buffer` produce a
  //! `cuda::device_buffer` from this resource.
  friend constexpr void get_property(const locality_domain_resource_ref&, ::cuda::mr::device_accessible) noexcept {}

  using default_queries = ::cuda::mr::properties_list<::cuda::mr::device_accessible>;

private:
  cuda::device_ref device_;
  unsigned int domain_;
};

//! Creates and owns a memory pool localized to one locality domain of one device.
//!
//! Non-copyable and non-movable, so the pool handle has exactly one owner. Pass `ref()` to
//! `make_buffer`; the resulting buffers copy the ref, not the owner.
class locality_domain_resource : public locality_domain_resource_ref
{
public:
  locality_domain_resource(cuda::device_ref device, unsigned int domain)
      : locality_domain_resource_ref{locality::create_domain_pool(device, domain), device, domain}
  {
    if (device.peers().size())
    {
      enable_access_from(device.peers());
    }
  }

  locality_domain_resource(const locality_domain_resource&)            = delete;
  locality_domain_resource& operator=(const locality_domain_resource&) = delete;
  locality_domain_resource(locality_domain_resource&&)                 = delete;
  locality_domain_resource& operator=(locality_domain_resource&&)      = delete;

  ~locality_domain_resource()
  {
    // The pool is torn down at process exit alongside every buffer drawn from it, so a failure here
    // is not actionable; swallow it rather than throwing from a destructor.
    _CCCL_ASSERT_CUDA_API(::cuda::__driver::__mempoolDestroyNoThrow, "Failed to destroy a memory pool", get());
  }

  [[nodiscard]] locality_domain_resource_ref ref() const noexcept
  {
    return *this;
  }
};

static_assert(::cuda::mr::resource_with<locality_domain_resource_ref, ::cuda::mr::device_accessible>);
} // namespace mgmn
