// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_reduce.cuh>

#include <cuda/__event/event.h>
#include <cuda/__event/timed_event.h>
#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/buffer>
#include <cuda/std/execution>
#include <cuda/stream>

#include <array>
#include <cstddef>
#include <future>
#include <stdexcept>
#include <vector>

#include <nccl_device.h>

#include "common.hpp"
#include "green_context_support.hpp"
#include "nccl_support.hpp"

namespace
{
//! Per-rank scalar windows registered as NCCL symmetric memory plus the device communicator.
struct rank_windows
{
  ncclDevComm devcomm{};
  ncclWindow_t source{};
  ncclWindow_t destination{};
};

//! Terminal-epilogue hook for the device NCCL path. Each green context's final CUB reduction
//! publishes its local aggregate into its registered source window, then rank 0 fuses the two
//! rank-local aggregates with a single-element device-side reduce/copy, bracketed by LSA
//! barriers. No host-launched collective occurs in the timed interval.
struct device_nccl_epilogue
{
  float* local_aggregate{};
  ncclDevComm devcomm{};
  ncclWindow_t source{};
  ncclWindow_t destination{};

  _CCCL_DEVICE_API void operator()(float value) const
  {
    *local_aggregate = value;

    const ncclCoopThread cooperative = ncclCoopThread{};
    ncclLsaBarrierSession<ncclCoopThread> barrier{cooperative, devcomm, ncclTeamTagLsa{}, 0};
    barrier.sync(cooperative, cuda::memory_order_acq_rel);
    if (devcomm.rank == 0)
    {
      ncclLsaReduceSumCopy<float>(cooperative, source, 0, destination, 0, 1, ncclTeamLsa(devcomm));
    }
    barrier.sync(cooperative, cuda::memory_order_acq_rel);
  }
};

void benchmark_cudax_device_nccl(benchmark::State& state)
{
  const auto elements = static_cast<int>(state.range(0));
  const auto half     = elements / 2;

  mgmn::green_partition partition{cuda::devices[0]};
  const auto device = partition.device();

  cuda::stream coordinator{device};
  cuda::timed_event start{device};
  cuda::timed_event stop{device};
  std::array<cuda::event, mgmn::green_partition::rank_count> completed{cuda::event{device}, cuda::event{device}};

  // Each green context owns its input half, CUB destination, and NCCL scalar windows.
  const std::vector<float> half_values(static_cast<std::size_t>(half), 1.0F);
  std::array inputs{cuda::make_device_buffer<float>(partition.stream(0), device, half_values),
                    cuda::make_device_buffer<float>(partition.stream(1), device, half_values)};
  std::array outputs{cuda::make_device_buffer<float>(partition.stream(0), device, 1, cuda::no_init),
                     cuda::make_device_buffer<float>(partition.stream(1), device, 1, cuda::no_init)};
  std::array aggregates{cuda::make_device_buffer<float>(partition.stream(0), device, 1, cuda::no_init),
                        cuda::make_device_buffer<float>(partition.stream(1), device, 1, cuda::no_init)};
  std::array destinations{cuda::make_device_buffer<float>(partition.stream(0), device, 1, cuda::no_init),
                          cuda::make_device_buffer<float>(partition.stream(1), device, 1, cuda::no_init)};
  partition.stream(0).sync();
  partition.stream(1).sync();

  // Initialize host and device communicators per green context, registering the rank-local
  // aggregate and destination scalars as symmetric windows. ncclCommInitRank and
  // ncclDevCommCreate are collective, so each rank owns its host thread with its context current.
  ncclUniqueId unique_id{};
  mgmn::check_nccl(ncclGetUniqueId(&unique_id), "ncclGetUniqueId");
  std::array<ncclComm_t, mgmn::green_partition::rank_count> host_communicators{};
  std::array<rank_windows, mgmn::green_partition::rank_count> windows{};

  const auto setup_rank = [&](int rank) {
    cuda::__ensure_current_context guard{partition.context(rank).__transformed};
    mgmn::check_nccl(ncclCommInitRank(&host_communicators[rank], 2, unique_id, rank), "ncclCommInitRank");

    ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
    mgmn::check_nccl(ncclCommQueryProperties(host_communicators[rank], &properties), "ncclCommQueryProperties");
    if (!properties.deviceApiSupport)
    {
      throw std::runtime_error("NCCL device API is unavailable on this build");
    }

    mgmn::check_nccl(
      ncclCommWindowRegister(
        host_communicators[rank],
        aggregates[rank].data(),
        sizeof(float),
        &windows[rank].source,
        NCCL_WIN_COLL_SYMMETRIC),
      "ncclCommWindowRegister(source)");
    mgmn::check_nccl(
      ncclCommWindowRegister(
        host_communicators[rank],
        destinations[rank].data(),
        sizeof(float),
        &windows[rank].destination,
        NCCL_WIN_COLL_SYMMETRIC),
      "ncclCommWindowRegister(destination)");

    ncclDevCommRequirements_t requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    requirements.lsaBarrierCount           = 1;
    mgmn::check_nccl(ncclDevCommCreate(host_communicators[rank], &requirements, &windows[rank].devcomm),
                     "ncclDevCommCreate");
  };

  std::array<std::future<void>, mgmn::green_partition::rank_count> initialization;
  for (int rank = 0; rank != mgmn::green_partition::rank_count; ++rank)
  {
    initialization[rank] = std::async(std::launch::async, setup_rank, rank);
  }
  for (auto& task : initialization)
  {
    task.get();
  }

  for (auto _ : state)
  {
    static_cast<void>(_);
    mgmn::begin_partition_timing(partition, coordinator, start);
    for (int rank = 0; rank != mgmn::green_partition::rank_count; ++rank)
    {
      const auto epilogue = device_nccl_epilogue{
        aggregates[rank].data(), windows[rank].devcomm, windows[rank].source, windows[rank].destination};
      const auto environment =
        cuda::std::execution::env{cuda::stream_ref{partition.stream(rank)}, cub::terminal_epilogue(epilogue)};
      _CCCL_TRY_CUDA_API(
        cub::DeviceReduce::Reduce,
        "Device-NCCL terminal-epilogue reduction failed",
        inputs[rank].data(),
        outputs[rank].data(),
        half,
        cuda::std::plus<>{},
        0.0F,
        environment);
    }
    mgmn::end_partition_timing(partition, coordinator, start, stop, completed, state);
  }

  const auto teardown_rank = [&](int rank) {
    cuda::__ensure_current_context guard{partition.context(rank).__transformed};
    mgmn::check_nccl(ncclDevCommDestroy(host_communicators[rank], &windows[rank].devcomm), "ncclDevCommDestroy");
    mgmn::check_nccl(ncclCommWindowDeregister(host_communicators[rank], windows[rank].source),
                     "ncclCommWindowDeregister(source)");
    mgmn::check_nccl(ncclCommWindowDeregister(host_communicators[rank], windows[rank].destination),
                     "ncclCommWindowDeregister(destination)");
    mgmn::check_nccl(ncclCommDestroy(host_communicators[rank]), "ncclCommDestroy");
  };
  std::array<std::future<void>, mgmn::green_partition::rank_count> teardown;
  for (int rank = 0; rank != mgmn::green_partition::rank_count; ++rank)
  {
    teardown[rank] = std::async(std::launch::async, teardown_rank, rank);
  }
  for (auto& task : teardown)
  {
    task.get();
  }
  mgmn::set_common_counters(state, static_cast<std::size_t>(elements), partition.sm_count());
}
} // namespace

int main(int argc, char** argv)
{
  return mgmn::run_benchmark(argc, argv, "cudax_device_nccl", benchmark_cudax_device_nccl);
}
