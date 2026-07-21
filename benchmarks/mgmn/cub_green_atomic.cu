// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_reduce.cuh>

#include <cuda/__driver/driver_api.h>
#include <cuda/__event/event.h>
#include <cuda/__event/timed_event.h>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/std/execution>
#include <cuda/stream>

#include <cuda/experimental/green_context.cuh>
#include <cuda/experimental/stream.cuh>

#include <array>
#include <cstddef>
#include <vector>

#include <cuda.h>

#include "common.hpp"

namespace cudax = cuda::experimental;

namespace
{
//! Resolve a versioned driver entry point by name. Mirrors the `_CCCLRT_GET_DRIVER_FUNCTION_VERSIONED`
//! macro in `cuda/__driver/driver_api.h`, which is `#undef`-ed there and so is not visible here.
#define MGMN_GET_DRIVER_FUNCTION_VERSIONED(function_name, major, minor) \
  reinterpret_cast<decltype(::function_name)*>(::cuda::__driver::__get_driver_entry_point(#function_name, major, minor))

//! File-local driver stubs for the green-context split APIs that are not yet wrapped in
//! `cuda::__driver`. Each mirrors the existing stubs there: fetch the versioned entry point
//! once and route the call through `cuda::__driver::__call_driver_fn` for error handling.
[[nodiscard]] CUdevResource green_sm_resource(CUcontext context)
{
  CUdevResource result{};
  static auto driver_fn = MGMN_GET_DRIVER_FUNCTION_VERSIONED(cuCtxGetDevResource, 12, 5);
  ::cuda::__driver::__call_driver_fn(
    driver_fn, "Failed to query the SM resource of a context", context, &result, CU_DEV_RESOURCE_TYPE_SM);
  return result;
}

//! Split `input` into groups of exactly `sms_per_group` SMs. Returns the produced groups; the
//! remainder is discarded. The two-call protocol first queries the group count, then fills them.
[[nodiscard]] std::vector<CUdevResource> split_sm_resource(const CUdevResource& input, unsigned int sms_per_group)
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
[[nodiscard]] CUdevResourceDesc make_resource_desc(CUdevResource& group)
{
  CUdevResourceDesc descriptor{};
  static auto driver_fn = MGMN_GET_DRIVER_FUNCTION_VERSIONED(cuDevResourceGenerateDesc, 12, 5);
  ::cuda::__driver::__call_driver_fn(
    driver_fn, "Failed to generate a green-context resource descriptor", &descriptor, &group, 1u);
  return descriptor;
}

//! Create a green context over the SMs described by `descriptor`.
[[nodiscard]] CUgreenCtx create_green_ctx(CUdevResourceDesc descriptor, CUdevice device)
{
  CUgreenCtx green_ctx{};
  static auto driver_fn = MGMN_GET_DRIVER_FUNCTION_VERSIONED(cuGreenCtxCreate, 12, 5);
  ::cuda::__driver::__call_driver_fn(
    driver_fn, "Failed to create a green context", &green_ctx, descriptor, device, CU_GREEN_CTX_DEFAULT_STREAM);
  return green_ctx;
}

//! Create a non-blocking stream that submits into `green_ctx`.
[[nodiscard]] cudaStream_t create_green_ctx_stream(CUgreenCtx green_ctx)
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
[[nodiscard]] std::vector<cudax::green_context> make_green_halves(cuda::device_ref device, int partitions)
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

//! Terminal-epilogue hook: each green context's final CUB reduction adds its local aggregate
//! into one device-scope location with a single relaxed floating-point atomic.
struct atomic_epilogue
{
  float* aggregate{};

  _CCCL_DEVICE_API void operator()(float value) const
  {
    ::cuda::atomic_ref<float, ::cuda::thread_scope_device>{*aggregate}.fetch_add(value, ::cuda::memory_order_relaxed);
  }
};

void benchmark_cub_green_atomic(benchmark::State& state)
{
  constexpr int rank_count = 2;
  const auto elements      = static_cast<std::size_t>(state.range(0));
  const auto device        = cuda::devices[0];

  // Split the device into equal green-context halves and give each its own non-blocking stream
  // created directly against the green context via `cuGreenCtxStreamCreate`.
  const auto contexts = make_green_halves(device, rank_count);

  std::vector<cuda::stream> streams;

  streams.reserve(rank_count);
  for (int rank = 0; rank < rank_count; ++rank)
  {
    streams.emplace_back(cuda::stream::from_native_handle(create_green_ctx_stream(contexts[rank].__green_ctx)));
  }

  std::vector<cuda::device_buffer<float>> inputs;
  std::vector<cuda::device_buffer<float>> local_outputs;

  auto aggregate = cuda::make_device_buffer<float>(cuda::stream_ref{cudaStream_t{}}, device, 1, cuda::no_init);

  using env_type = decltype(cuda::std::execution::env{
    cuda::stream_ref{streams[0]}, cub::terminal_epilogue(atomic_epilogue{aggregate.data()})});

  std::vector<env_type> envs;

  inputs.reserve(rank_count);
  local_outputs.reserve(rank_count);
  envs.reserve(rank_count);

  for (int rank = 0; rank < rank_count; ++rank)
  {
    inputs.emplace_back(cuda::make_device_buffer<float>(streams[rank], device, elements / rank_count, 1.0F));
    local_outputs.emplace_back(cuda::make_device_buffer<float>(streams[rank], device, 1, cuda::no_init));
    envs.emplace_back(cuda::std::execution::env{
      cuda::stream_ref{streams[rank]}, cub::terminal_epilogue(atomic_epilogue{aggregate.data()})});
  }

  cuda::timed_event start{device};
  cuda::timed_event stop{device};
  cuda::event completed{device};

  for (auto&& s : streams)
  {
    s.sync();
  }

  for (auto _ : state)
  {
    static_cast<void>(_);
    // Establish a common start boundary: record `start` on stream 0 and make stream 1 wait on it.
    start.record(streams.front());
    for (int rank = 1; rank < rank_count; ++rank)
    {
      streams[rank].wait(start);
    }
    for (int rank = 0; rank != rank_count; ++rank)
    {
      _CCCL_TRY_CUDA_API(
        cub::DeviceReduce::Reduce,
        "Terminal-epilogue CUB reduction failed",
        inputs[rank].data(),
        local_outputs[rank].data(),
        inputs[rank].size(),
        cuda::std::plus<>{},
        0.0F,
        envs[rank]);
    }
    // Join both halves onto stream 0, then record and time the stop boundary.
    completed.record(streams.back());
    for (int rank = 0; rank < rank_count - 1; ++rank)
    {
      streams[rank].wait(completed);
    }
    stop.record(streams.front());
    stop.sync();
    state.SetIterationTime(static_cast<double>((stop - start).count()) / 1'000'000'000.0);
  }
  const auto sm_count = ::cuda::__driver::__deviceGetAttribute(
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, ::cuda::__driver::__deviceGet(device.get()));
  mgmn::set_common_counters(state, elements, static_cast<unsigned int>(sm_count));
}
} // namespace

int main(int argc, char** argv)
{
  return mgmn::run_benchmark(argc, argv, "cub_green_atomic", benchmark_cub_green_atomic);
}
