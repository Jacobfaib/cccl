// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cuda/__driver/driver_api.h>
#include <cuda/devices>

#include <cuda/experimental/green_context.cuh>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <cuda.h>

namespace cudax = cuda::experimental;

namespace mgmn
{
//! Resolve a versioned driver entry point by name. Mirrors the `_CCCLRT_GET_DRIVER_FUNCTION_VERSIONED`
//! macro in `cuda/__driver/driver_api.h`, which is `#undef`-ed there and so is not visible here.
#define MGMN_GET_DRIVER_FUNCTION_VERSIONED(function_name, major, minor) \
  reinterpret_cast<decltype(::function_name)*>(::cuda::__driver::__get_driver_entry_point(#function_name, major, minor))

//! File-local driver stubs for the green-context split APIs that are not yet wrapped in
//! `cuda::__driver`. Each mirrors the existing stubs there: fetch the versioned entry point
//! once and route the call through `cuda::__driver::__call_driver_fn` for error handling.
[[nodiscard]] inline CUdevResource green_sm_resource(CUcontext context)
{
  CUdevResource result{};
  static auto driver_fn = MGMN_GET_DRIVER_FUNCTION_VERSIONED(cuCtxGetDevResource, 12, 5);
  ::cuda::__driver::__call_driver_fn(
    driver_fn, "Failed to query the SM resource of a context", context, &result, CU_DEV_RESOURCE_TYPE_SM);
  return result;
}

//! Split `input` into groups of exactly `sms_per_group` SMs. Returns the produced groups; the
//! remainder is discarded. The two-call protocol first queries the group count, then fills them.
[[nodiscard]] inline std::vector<CUdevResource> split_sm_resource(const CUdevResource& input, unsigned int sms_per_group)
{
  static auto driver_fn = MGMN_GET_DRIVER_FUNCTION_VERSIONED(cuDevSmResourceSplitByCount, 12, 5);

  CUdevResource mutable_input = input;
  unsigned int group_count    = 0;
  ::cuda::__driver::__call_driver_fn(
    driver_fn,
    "Failed to query the green-context SM split group count",
    nullptr,
    &group_count,
    &mutable_input,
    nullptr,
    0U,
    sms_per_group);

  std::vector<CUdevResource> groups(group_count);
  CUdevResource remainder{};
  ::cuda::__driver::__call_driver_fn(
    driver_fn,
    "Failed to split the SM resource for green contexts",
    groups.data(),
    &group_count,
    &mutable_input,
    &remainder,
    0U,
    sms_per_group);
  groups.resize(group_count);
  return groups;
}

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

//! Split `device` into `partitions` green contexts, each over an equal, aligned share of the
//! device's SMs. Throws unless the SMs divide evenly and each share satisfies the driver's
//! minimum-size and coscheduling-alignment constraints, so every returned context covers exactly
//! one `1/partitions` slice.
[[nodiscard]] inline std::vector<cudax::green_context> make_green_halves(cuda::device_ref device, int partitions)
{
  device.init();
  const CUdevice cu_device = ::cuda::__driver::__deviceGet(device.get());

  // Query the device's full SM resource through its retained primary context.
  CUcontext primary        = ::cuda::__driver::__primaryCtxRetain(cu_device);
  const CUdevResource full = green_sm_resource(primary);
  static_cast<void>(::cuda::__driver::__primaryCtxReleaseNoThrow(cu_device));

  const unsigned int sm_count = full.sm.smCount;
  const unsigned int per_part = sm_count / static_cast<unsigned int>(partitions);
  if (partitions < 1 || sm_count % static_cast<unsigned int>(partitions) != 0 || per_part < full.sm.minSmPartitionSize
      || per_part % full.sm.smCoscheduledAlignment != 0)
  {
    throw std::runtime_error("the GPU cannot provide an exact aligned SM split into the requested partitions");
  }

  std::vector<CUdevResource> groups = split_sm_resource(full, per_part);
  if (static_cast<int>(groups.size()) < partitions)
  {
    throw std::runtime_error("cuDevSmResourceSplitByCount produced fewer groups than requested");
  }

  std::vector<cudax::green_context> contexts;
  contexts.reserve(static_cast<std::size_t>(partitions));
  for (int rank = 0; rank != partitions; ++rank)
  {
    if (groups[rank].sm.smCount != per_part)
    {
      throw std::runtime_error("cuDevSmResourceSplitByCount did not produce equal SM groups");
    }
    const CUdevResourceDesc descriptor = make_resource_desc(groups[rank]);
    const CUgreenCtx green_ctx         = create_green_ctx(descriptor, cu_device);
    contexts.push_back(cudax::green_context::from_native_handle(green_ctx));
  }
  return contexts;
}
} // namespace mgmn
