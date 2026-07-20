// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cuda/__event/event.h>
#include <cuda/__event/timed_event.h>
#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/buffer>
#include <cuda/std/execution>
#include <cuda/std/span>
#include <cuda/stream>

#include <cuda/experimental/__multi_gpu/algorithm/reduce/reduce.h>
#include <cuda/experimental/__multi_gpu/nccl_communicator.h>

#include <array>
#include <cstddef>
#include <future>
#include <vector>

#include "common.hpp"
#include "green_context_support.hpp"
#include "nccl_support.hpp"

namespace
{
//! Split the input across two green contexts and reduce with `cudax::reduce`, which performs a
//! local CUB reduction per rank followed by a host-launched NCCL collective.
void benchmark_cudax_host_nccl(benchmark::State& state)
{
  const auto elements = static_cast<int>(state.range(0));
  const auto half     = static_cast<std::size_t>(elements / 2);

  mgmn::green_partition partition{cuda::devices[0]};
  const auto device = partition.device();

  cuda::stream coordinator{device};
  cuda::timed_event start{device};
  cuda::timed_event stop{device};
  std::array<cuda::event, mgmn::green_partition::rank_count> completed{cuda::event{device}, cuda::event{device}};

  // Each green context owns its input half and its output scalar.
  const std::vector<float> half_values(half, 1.0F);
  std::array inputs_buf{cuda::make_device_buffer<float>(partition.stream(0), device, half_values),
                        cuda::make_device_buffer<float>(partition.stream(1), device, half_values)};
  std::array outputs{cuda::make_device_buffer<float>(partition.stream(0), device, 1, cuda::no_init),
                     cuda::make_device_buffer<float>(partition.stream(1), device, 1, cuda::no_init)};
  partition.stream(0).sync();
  partition.stream(1).sync();

  // Initialize one NCCL rank per green context. ncclCommInitRank is collective and blocking, so
  // each rank must own its host thread with its green context current.
  ncclUniqueId unique_id{};
  mgmn::check_nccl(ncclGetUniqueId(&unique_id), "ncclGetUniqueId");
  std::array<ncclComm_t, mgmn::green_partition::rank_count> native_communicators{};
  std::array<std::future<void>, mgmn::green_partition::rank_count> initialization;
  for (int rank = 0; rank != mgmn::green_partition::rank_count; ++rank)
  {
    initialization[rank] = std::async(std::launch::async, [&, rank] {
      cuda::__ensure_current_context guard{partition.context(rank).__transformed};
      mgmn::check_nccl(ncclCommInitRank(&native_communicators[rank], 2, unique_id, rank), "ncclCommInitRank");
    });
  }
  for (auto& task : initialization)
  {
    task.get();
  }

  std::vector<cudax::nccl_communicator> communicators;
  communicators.reserve(mgmn::green_partition::rank_count);
  for (int rank = 0; rank != mgmn::green_partition::rank_count; ++rank)
  {
    communicators.emplace_back(
      cudax::nccl_communicator::from_native_handle(native_communicators[rank], partition.logical_device(rank)));
  }

  const std::array inputs{cuda::std::span<float>{inputs_buf[0].data(), half},
                          cuda::std::span<float>{inputs_buf[1].data(), half}};
  const std::array environments{cuda::std::execution::env{cuda::stream_ref{partition.stream(0)}},
                                cuda::std::execution::env{cuda::stream_ref{partition.stream(1)}}};
  const std::array output_iterators{outputs[0].data(), outputs[1].data()};

  for (auto _ : state)
  {
    static_cast<void>(_);
    mgmn::begin_partition_timing(partition, coordinator, start);
    cudax::reduce(cudax::broadcasted, communicators, environments, inputs, output_iterators);
    mgmn::end_partition_timing(partition, coordinator, start, stop, completed, state);
  }
  mgmn::set_common_counters(state, static_cast<std::size_t>(elements), partition.sm_count());
}
} // namespace

int main(int argc, char** argv)
{
  return mgmn::run_benchmark(argc, argv, "cudax_host_nccl", benchmark_cudax_host_nccl);
}
