// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cuda/__driver/driver_api.h>
#include <cuda/__event/event.h>
#include <cuda/__event/timed_event.h>
#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/buffer>
#include <cuda/std/execution>
#include <cuda/stream>

#include <cuda/experimental/__device/logical_device.cuh>
#include <cuda/experimental/__multi_gpu/algorithm/reduce/reduce.h>
#include <cuda/experimental/__multi_gpu/nccl_communicator.h>
#include <cuda/experimental/stream.cuh>

#include <cstddef>
#include <future>
#include <vector>

#include "common.hpp"
#include "green_ctx_partition.hpp"
#include "nccl_support.hpp"

namespace
{
//! Split the input across `rank_count` green contexts and reduce with `cudax::reduce`, which
//! performs a local CUB reduction per rank followed by a host-launched NCCL collective.
void benchmark_cudax_host_nccl(benchmark::State& state)
{
  constexpr int rank_count = 2;
  const auto elements      = static_cast<int>(state.range(0));
  const auto per_rank      = static_cast<std::size_t>(elements) / rank_count;
  const auto device        = cuda::devices[0];

  // Split the device into equal green-context partitions and give each its own non-blocking
  // stream created directly against the green context via `cuGreenCtxStreamCreate`.
  const auto contexts = mgmn::make_green_halves(device, rank_count);

  std::vector<cuda::stream> streams;
  streams.reserve(rank_count);
  for (int rank = 0; rank < rank_count; ++rank)
  {
    streams.emplace_back(cuda::stream::from_native_handle(mgmn::create_green_ctx_stream(contexts[rank].__green_ctx)));
  }

  // Initialize one NCCL rank per green context. ncclCommInitRank is collective and blocking, so
  // each rank must own its host thread with its green context current.
  ncclUniqueId unique_id{};
  mgmn::check_nccl(ncclGetUniqueId(&unique_id), "ncclGetUniqueId");

  std::vector<cudax::nccl_communicator> communicators;

  communicators.reserve(rank_count);
  std::generate_n(std::back_inserter(communicators), rank_count, [] {
    return cudax::nccl_communicator{cuda::no_init};
  });

  {
    std::vector<std::future<void>> initialization(rank_count);
    for (int rank = 0; rank != rank_count; ++rank)
    {
      initialization[rank] = std::async(std::launch::async, [&, rank] {
        ncclComm_t comm;

        cuda::__ensure_current_context guard{contexts[rank].__transformed};
        mgmn::check_nccl(ncclCommInitRank(&comm, rank_count, unique_id, rank), "ncclCommInitRank");

        communicators[rank] = cudax::nccl_communicator::from_native_handle(comm, cudax::logical_device{contexts[rank]});
      });
    }
    for (auto& task : initialization)
    {
      task.get();
    }
  }

  std::vector<decltype(cuda::std::execution::env{cuda::stream_ref{streams[0]}})> environments;
  environments.reserve(rank_count);
  for (int rank = 0; rank < rank_count; ++rank)
  {
    environments.emplace_back(cuda::std::execution::env{cuda::stream_ref{streams[rank]}});
  }

  cuda::timed_event start{device};
  cuda::timed_event stop{device};
  cuda::event completed{device};

  // Each green context owns its input half and its output scalar.
  std::vector<cuda::device_buffer<float>> inputs_buf;
  std::vector<cuda::device_buffer<float>> outputs;
  std::vector<cuda::device_buffer<float>::iterator> output_its;

  inputs_buf.reserve(rank_count);
  outputs.reserve(rank_count);
  output_its.reserve(rank_count);
  for (int rank = 0; rank < rank_count; ++rank)
  {
    inputs_buf.emplace_back(cuda::make_device_buffer<float>(streams[rank], device, per_rank, 1.0F));
    auto& o = outputs.emplace_back(cuda::make_device_buffer<float>(streams[rank], device, 1, cuda::no_init));
    output_its.emplace_back(o.begin());
  }
  for (auto&& s : streams)
  {
    s.sync();
  }

  for (auto _ : state)
  {
    static_cast<void>(_);
    // Establish a common start boundary: record `start` on stream 0 and make the rest wait on it.
    start.record(streams.front());
    for (int rank = 1; rank < rank_count; ++rank)
    {
      streams[rank].wait(start);
    }
    cudax::reduce(cudax::broadcasted, communicators, environments, inputs_buf, output_its);
    // Join every partition onto stream 0, then record and time the stop boundary.
    for (int rank = 1; rank < rank_count; ++rank)
    {
      completed.record(streams[rank]);
      streams.front().wait(completed);
    }
    stop.record(streams.front());
    stop.sync();
    state.SetIterationTime(static_cast<double>((stop - start).count()) / 1'000'000'000.0);
  }

  const auto sm_count = ::cuda::__driver::__deviceGetAttribute(
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, ::cuda::__driver::__deviceGet(device.get()));
  mgmn::set_common_counters(state, static_cast<std::size_t>(elements), static_cast<unsigned int>(sm_count));
}
} // namespace

int main(int argc, char** argv)
{
  return mgmn::run_benchmark(argc, argv, "cudax_host_nccl", benchmark_cudax_host_nccl);
}
