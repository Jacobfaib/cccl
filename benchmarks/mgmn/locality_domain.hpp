// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

//! Bare CUDA driver calls for locality domains (CUDA 13.4+).
//!
//! A locality domain is a hardware partition of a GPU with its own affinity between a subset of the
//! SMs and a subset of the memory. Placing both the execution and the data of a rank in the same
//! domain keeps that rank's traffic local instead of crossing the on-package interconnect.
//!
//! Two independent driver mechanisms are needed and they must agree on the domain id:
//!   - execution: `cuDevSmResourceSplit` with `CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID`, which
//!     yields the SM group belonging to a domain, wrapped in a green context;
//!   - memory: a memory pool created with `CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN`, whose
//!     backing store is placed in that same domain.
//!
//! Domain index `i` is the same `localityDomainId` for both, so exec(i) and data(i) are co-located.
//!
//! These entry points ship in CUDA 13.4, which is newer than the toolkit this file may be compiled
//! against. Everything below is therefore resolved dynamically by name through the driver entry
//! point API and declared locally, so this header compiles against an older `cuda.h` and fails at
//! run time (not build time) when the driver is too old.

#include <cuda/__driver/driver_api.h>
#include <cuda/devices>

#include <cuda/experimental/green_context.cuh>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>

namespace cudax = cuda::experimental;

namespace mgmn::locality
{
//! Resolve a driver entry point by name at the given ABI version. Mirrors the
//! `_CCCLRT_GET_DRIVER_FUNCTION_VERSIONED` macro in `cuda/__driver/driver_api.h`, which is
//! `#undef`-ed there and so is not reachable from here. Unlike that macro, the symbol is named by
//! string rather than by `decltype`, because these symbols need not exist in the toolkit headers.
template <typename FnT>
[[nodiscard]] inline FnT get_driver_function(const char* name, int major, int minor)
{
  return reinterpret_cast<FnT>(::cuda::__driver::__get_driver_entry_point(name, major, minor));
}

//! Number of locality domains on `device`, or 0 when the driver predates the attribute.
[[nodiscard]] inline unsigned int domain_count(cuda::device_ref device)
{
  int count = 0;
  if (::cuDeviceGetAttribute(&count, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, device.get()) != CUDA_SUCCESS)
  {
    return 0;
  }
  return count > 0 ? static_cast<unsigned int>(count) : 0;
}

//! Split the device's full SM resource into one group per locality domain.
//!
//! Each `groupParams` entry selects a domain by id rather than by SM count, so group `i` is exactly
//! the set of SMs that belong to locality domain `i`. `smCount == 0` puts the call in discovery
//! mode: the driver fills in however many SMs that domain actually has.
[[nodiscard]] inline std::vector<CUdevResource> split_by_domain(cuda::device_ref device, unsigned int domains)
{
  using split_fn_t = CUresult(CUDAAPI*)(
    CUdevResource*, unsigned int, const CUdevResource*, CUdevResource*, unsigned int, CU_DEV_SM_RESOURCE_GROUP_PARAMS*);
  static auto driver_fn = get_driver_function<split_fn_t>("cuDevSmResourceSplit", 13, 4);

  CUdevResource full{};
  ::cuda::__driver::__call_driver_fn(
    ::cuDeviceGetDevResource, "Failed to query the device SM resource", device.get(), &full, CU_DEV_RESOURCE_TYPE_SM);

  std::vector<CU_DEV_SM_RESOURCE_GROUP_PARAMS> params(domains);
  for (unsigned int domain = 0; domain < domains; ++domain)
  {
    params[domain].flags            = CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID;
    params[domain].localityDomainId = domain;
  }

  std::vector<CUdevResource> groups(domains);
  CUdevResource remainder{};
  ::cuda::__driver::__call_driver_fn(
    driver_fn,
    "Failed to split the SM resource by locality domain",
    groups.data(),
    domains,
    &full,
    &remainder,
    0U,
    params.data());
  return groups;
}

//! Create a memory pool whose backing store lives in locality domain `domain` of `device`.
//! Allocations from this pool are stream-ordered (`cuMemAllocFromPoolAsync`).
[[nodiscard]] inline CUmemoryPool create_domain_pool(cuda::device_ref device, unsigned int domain)
{
  using create_fn_t     = CUresult(CUDAAPI*)(CUmemoryPool*, const CUmemPoolProps*);
  static auto driver_fn = get_driver_function<create_fn_t>("cuMemPoolCreate", 13, 4);

  CUmemPoolProps props{};
  // Exportable handles are a precondition for registering allocations from this pool as NCCL
  // symmetric memory windows. Both types are requested: POSIX file descriptors cover
  // single-process multi-rank on one node, while fabric handles are what a multi-node MNNVL setup
  // needs and require IMEX to be configured.
  props.handleTypes =
    static_cast<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR | CU_MEM_HANDLE_TYPE_FABRIC);
  props.allocType                           = CU_MEM_ALLOCATION_TYPE_PINNED;
  props.location.type                       = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
  props.location.localized.deviceId         = static_cast<unsigned char>(device.get());
  props.location.localized.localityDomainId = static_cast<unsigned char>(domain);

  CUmemoryPool pool{};
  ::cuda::__driver::__call_driver_fn(driver_fn, "Failed to create a locality-domain memory pool", &pool, &props);

  ::cuda::std::size_t thres = ::cuda::std::numeric_limits<size_t>::max();
  ::cuda::__driver::__mempoolSetAttribute(pool, ::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &thres);
  return pool;
}

//! Locality domain a device pointer's backing store resides in. Used to verify that an allocation
//! actually landed where it was asked to.
[[nodiscard]] inline unsigned int pointer_domain(void* ptr)
{
  unsigned int ordinal = ~0U;
  ::cuda::__driver::__call_driver_fn(
    ::cuPointerGetAttribute,
    "Failed to query the locality domain of a pointer",
    &ordinal,
    CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL,
    reinterpret_cast<CUdeviceptr>(ptr));
  return ordinal;
}
} // namespace mgmn::locality

namespace mgmn
{
//! Resolve a versioned driver entry point by name. Mirrors the `_CCCLRT_GET_DRIVER_FUNCTION_VERSIONED`
//! macro in `cuda/__driver/driver_api.h`, which is `#undef`-ed there and so is not visible here.
//! Unlike `locality::get_driver_function`, the symbol is named by `decltype`, which is possible
//! because these green-context entry points predate 13.4 and so exist in the toolkit headers.
#define MGMN_GET_DRIVER_FUNCTION_VERSIONED(function_name, major, minor) \
  reinterpret_cast<decltype(::function_name)*>(::cuda::__driver::__get_driver_entry_point(#function_name, major, minor))

//! File-local driver stubs for the green-context APIs that are not yet wrapped in
//! `cuda::__driver`. Each mirrors the existing stubs there: fetch the versioned entry point
//! once and route the call through `cuda::__driver::__call_driver_fn` for error handling.
//!
//! Wrap a single SM resource group into a green-context resource descriptor.
[[nodiscard]] inline CUdevResourceDesc make_resource_desc(CUdevResource& group)
{
  CUdevResourceDesc descriptor{};
  static auto driver_fn = MGMN_GET_DRIVER_FUNCTION_VERSIONED(cuDevResourceGenerateDesc, 12, 5);
  ::cuda::__driver::__call_driver_fn(
    driver_fn, "Failed to generate a green-context resource descriptor", &descriptor, &group, 1U);
  return descriptor;
}

//! Create a green context over the SMs described by `descriptor`.
[[nodiscard]] inline CUgreenCtx create_green_ctx(CUdevResourceDesc descriptor, CUdevice device)
{
  CUgreenCtx green_ctx{};
  static auto driver_fn = MGMN_GET_DRIVER_FUNCTION_VERSIONED(cuGreenCtxCreate, 12, 5);
  ::cuda::__driver::__call_driver_fn(
    driver_fn, "Failed to create a green context", &green_ctx, descriptor, device, CU_GREEN_CTX_DEFAULT_STREAM);
  return green_ctx;
}

//! Create a non-blocking stream that submits into `green_ctx`.
[[nodiscard]] inline cudaStream_t create_green_ctx_stream(CUgreenCtx green_ctx)
{
  CUstream stream{};
  static auto driver_fn = MGMN_GET_DRIVER_FUNCTION_VERSIONED(cuGreenCtxStreamCreate, 12, 5);
  ::cuda::__driver::__call_driver_fn(
    driver_fn, "Failed to create a green-context stream", &stream, green_ctx, CU_STREAM_NON_BLOCKING, 0);
  return stream;
}

#undef MGMN_GET_DRIVER_FUNCTION_VERSIONED

//! Build one green context per locality domain of `device`.
//!
//! Each SM group is selected by locality domain id rather than by SM count, so green context `i`
//! owns exactly the SMs of domain `i` and pairs with a `locality_domain_resource` for the same `i`.
//! Splitting by count instead would produce partitions with no relationship to the hardware's
//! memory affinity, which defeats the purpose of localizing the data.
[[nodiscard]] inline std::vector<cudax::green_context> make_domain_contexts(cuda::device_ref device, int domains)
{
  device.init();

  std::vector<CUdevResource> groups = locality::split_by_domain(device, static_cast<unsigned int>(domains));

  std::vector<cudax::green_context> contexts;
  contexts.reserve(static_cast<std::size_t>(domains));
  for (int domain = 0; domain != domains; ++domain)
  {
    if (groups[domain].sm.smCount == 0)
    {
      throw std::runtime_error("a locality domain was assigned no SMs");
    }
    const CUdevResourceDesc descriptor = make_resource_desc(groups[domain]);
    contexts.push_back(cudax::green_context::from_native_handle(create_green_ctx(descriptor, device.get())));
  }
  return contexts;
}
} // namespace mgmn
