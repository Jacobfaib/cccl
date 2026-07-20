// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cuda/__driver/driver_api.h>
#include <cuda/__event/event.h>
#include <cuda/__event/timed_event.h>
#include <cuda/devices>
#include <cuda/stream>

#include <cuda/experimental/green_context.cuh>
#include <cuda/experimental/stream.cuh>

#include <array>
#include <memory>
#include <stdexcept>
#include <string>

#include <cuda.h>

#include <benchmark/benchmark.h>

namespace cudax = cuda::experimental;

namespace mgmn
{
//! Throw on a failed CUDA driver call, formatting the driver error string.
inline void check_driver(CUresult status, const char* operation)
{
  if (status != CUDA_SUCCESS)
  {
    const char* error = "unknown CUDA driver error";
    static_cast<void>(::cuGetErrorString(status, &error));
    throw std::runtime_error(std::string{operation} + ": " + error);
  }
}

//! A device split into two equal green contexts, one host thread's worth of state per rank.
//!
//! Owns the two green contexts (via `cudax::green_context`, which is non-movable, hence the
//! `unique_ptr`) and a non-blocking stream on each. Rank timing is coordinated through a
//! primary-context stream so both partitions share a common start/stop boundary.
class green_partition
{
public:
  static constexpr int rank_count = 2;

  explicit green_partition(cuda::device_ref device)
      : device_{device}
  {
    device_.init();

    CUdevResource available{};
    check_driver(::cuDeviceGetDevResource(device_.get(), &available, CU_DEV_RESOURCE_TYPE_SM),
                 "cuDeviceGetDevResource(SM)");
    sm_count_                  = available.sm.smCount;
    const unsigned int half_sm = sm_count_ / 2;
    if (sm_count_ % 2 != 0 || half_sm < available.sm.minSmPartitionSize
        || half_sm % available.sm.smCoscheduledAlignment != 0)
    {
      throw std::runtime_error("the GPU cannot provide an exact aligned 50/50 SM split");
    }

    // Split the SM resource into two equal groups.
    CUdevResource groups[rank_count]{};
    CUdevResource remaining{};
    unsigned int group_count = rank_count;
    check_driver(::cuDevSmResourceSplitByCount(groups, &group_count, &available, &remaining, 0, half_sm),
                 "cuDevSmResourceSplitByCount");
    if (group_count != rank_count || groups[0].sm.smCount != half_sm || groups[1].sm.smCount != half_sm)
    {
      throw std::runtime_error("cuDevSmResourceSplitByCount did not produce an exact 50/50 SM split");
    }

    // Materialize a green context and a non-blocking stream for each group.
    const CUdevice cu_device = ::cuda::__driver::__deviceGet(device_.get());
    for (int rank = 0; rank != rank_count; ++rank)
    {
      CUdevResourceDesc descriptor{};
      check_driver(::cuDevResourceGenerateDesc(&descriptor, &groups[rank], 1), "cuDevResourceGenerateDesc");
      CUgreenCtx green_ctx{};
      check_driver(::cuGreenCtxCreate(&green_ctx, descriptor, cu_device, CU_GREEN_CTX_DEFAULT_STREAM),
                   "cuGreenCtxCreate");
      contexts_[rank] = std::make_unique<cudax::green_context>(cudax::green_context::from_native_handle(green_ctx));
      streams_[rank]  = cudax::stream{*contexts_[rank]};
    }
  }

  [[nodiscard]] cuda::device_ref device() const noexcept
  {
    return device_;
  }

  [[nodiscard]] unsigned int sm_count() const noexcept
  {
    return sm_count_;
  }

  [[nodiscard]] cudax::green_context& context(int rank) const noexcept
  {
    return *contexts_[rank];
  }

  [[nodiscard]] CUgreenCtx green_handle(int rank) const noexcept
  {
    return contexts_[rank]->__green_ctx;
  }

  [[nodiscard]] cudax::logical_device logical_device(int rank) const noexcept
  {
    return cudax::logical_device{*contexts_[rank]};
  }

  [[nodiscard]] cudax::stream& stream(int rank) noexcept
  {
    return streams_[rank];
  }

private:
  cuda::device_ref device_;
  unsigned int sm_count_{};
  std::array<std::unique_ptr<cudax::green_context>, rank_count> contexts_{};
  std::array<cudax::stream, rank_count> streams_{cudax::stream{cuda::no_init}, cudax::stream{cuda::no_init}};
};

//! Coordinated start boundary: record `start` on the coordinator stream and make both green
//! contexts wait on it before any partitioned work is submitted.
inline void begin_partition_timing(green_partition& partition, cuda::stream_ref coordinator, cuda::timed_event& start)
{
  start.record(coordinator);
  for (int rank = 0; rank != green_partition::rank_count; ++rank)
  {
    check_driver(::cuGreenCtxWaitEvent(partition.green_handle(rank), start.get()), "cuGreenCtxWaitEvent(start)");
  }
}

//! Coordinated stop boundary: record a completion event on each green context, join them on the
//! coordinator stream, then record and synchronize `stop`. Reports the elapsed start-to-stop
//! time through `SetIterationTime`.
inline void end_partition_timing(
  green_partition& partition,
  cuda::stream_ref coordinator,
  cuda::timed_event& start,
  cuda::timed_event& stop,
  std::array<cuda::event, green_partition::rank_count>& completed,
  benchmark::State& state)
{
  for (int rank = 0; rank != green_partition::rank_count; ++rank)
  {
    check_driver(::cuGreenCtxRecordEvent(partition.green_handle(rank), completed[rank].get()),
                 "cuGreenCtxRecordEvent(completed)");
    coordinator.wait(completed[rank]);
  }
  stop.record(coordinator);
  stop.sync();
  state.SetIterationTime(static_cast<double>((stop - start).count()) / 1'000'000'000.0);
}
} // namespace mgmn
