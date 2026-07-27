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
//! The 13.4 driver constants, declared locally because the toolkit's `cuda.h` may predate them.
//! The values are part of the driver ABI, so hard-coding them is safe for a benchmark; each is
//! validated at run time by the driver call that consumes it.
inline constexpr auto attribute_locality_domain_count              = static_cast<CUdevice_attribute>(147);
inline constexpr auto mem_location_type_locality_domain            = static_cast<CUmemLocationType>(0x6);
inline constexpr auto pointer_attribute_locality_domain            = static_cast<CUpointer_attribute>(24);
inline constexpr unsigned int sm_resource_group_locality_domain_id = 0x2;

//! Mirror of the 13.4 `CU_DEV_SM_RESOURCE_GROUP_PARAMS`. Layout must match the driver's, since a
//! pointer to an array of these is handed to `cuDevSmResourceSplit`.
struct sm_resource_group_params
{
  unsigned int smCount{};
  unsigned int coscheduledSmCount{};
  unsigned int preferredCoscheduledSmCount{};
  unsigned int flags{};
  unsigned int localityDomainId{};
  unsigned int reserved[11]{};
};

//! Mirror of the 13.4 `CUmemLocation`, whose `id` union gains a `localized` member carrying the
//! (device, locality domain) pair. Same size and layout as the driver's type.
struct mem_location
{
  CUmemLocationType type{};
  union
  {
    int id;
    struct
    {
      unsigned char deviceId;
      unsigned char localityDomainId;
      unsigned char reserved[2];
    } localized;
  };
};

//! Mirror of the 13.4 `CUmemPoolProps` with the extended location type. The trailing reserved bytes
//! keep the struct the size the driver expects.
struct mem_pool_props
{
  CUmemAllocationType allocType{};
  CUmemAllocationHandleType handleTypes{};
  mem_location location{};
  void* win32SecurityAttributes{};
  size_t maxSize{};
  unsigned short usage{};
  unsigned char reserved[54]{};
};

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
  if (::cuDeviceGetAttribute(&count, attribute_locality_domain_count, device.get()) != CUDA_SUCCESS)
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
    CUdevResource*, unsigned int, const CUdevResource*, CUdevResource*, unsigned int, sm_resource_group_params*);
  static auto driver_fn = get_driver_function<split_fn_t>("cuDevSmResourceSplit", 13, 4);

  CUdevResource full{};
  ::cuda::__driver::__call_driver_fn(
    ::cuDeviceGetDevResource, "Failed to query the device SM resource", device.get(), &full, CU_DEV_RESOURCE_TYPE_SM);

  std::vector<sm_resource_group_params> params(domains);
  for (unsigned int domain = 0; domain != domains; ++domain)
  {
    params[domain].flags            = sm_resource_group_locality_domain_id;
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
  using create_fn_t     = CUresult(CUDAAPI*)(CUmemoryPool*, const mem_pool_props*);
  static auto driver_fn = get_driver_function<create_fn_t>("cuMemPoolCreate", 13, 4);

  mem_pool_props props{};
  props.allocType                           = CU_MEM_ALLOCATION_TYPE_PINNED;
  props.location.type                       = mem_location_type_locality_domain;
  props.location.localized.deviceId         = static_cast<unsigned char>(device.get());
  props.location.localized.localityDomainId = static_cast<unsigned char>(domain);

  CUmemoryPool pool{};
  ::cuda::__driver::__call_driver_fn(driver_fn, "Failed to create a locality-domain memory pool", &pool, &props);
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
    pointer_attribute_locality_domain,
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
