// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// BEGIN TEMPORARY STRIPED PLACEMENT DIAGNOSTICS
// Delete the matching marked sections in this file and dispatch_reduce.cuh to remove the probes.
#include <cuda/__driver/driver_api.h>
#include <cuda/__runtime/api_wrapper.h>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/stream>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace striped_reduce_placement
{
inline thread_local bool requested = false;

struct block_interval
{
  unsigned long long start;
  unsigned long long end;
  unsigned sm;
};

template <class GridMapping>
struct timed_mapping : GridMapping
{
  block_interval* records;
  unsigned long long start_time{};
  unsigned sm{};

  _CCCL_DEVICE _CCCL_FORCEINLINE void record_start()
  {
    if (threadIdx.x == 0)
    {
      asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start_time) : : "memory");
    }
    __syncthreads();
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void record_end()
  {
    __syncthreads();
    if (threadIdx.x == 0)
    {
      unsigned long long end_time;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end_time) : : "memory");
      records[blockIdx.x] = {start_time, end_time, sm};
    }
  }
};

inline void report_overlap(const std::vector<block_interval>& records,
                           const std::unordered_map<unsigned, int>& sm_domains,
                           int device,
                           int domain_count)
{
  std::vector<std::vector<block_interval>> intervals(domain_count);
  auto origin         = std::numeric_limits<unsigned long long>::max();
  unsigned unknown    = 0;
  unsigned mismatches = 0;
  const auto capacity = (records.size() + domain_count - 1) / domain_count;
  for (cuda::std::size_t block = 0; block < records.size(); ++block)
  {
    const auto& record = records[block];
    if (record.end <= record.start)
    {
      throw std::runtime_error("Reduction diagnostic has an invalid block interval");
    }
    origin           = std::min(origin, record.start);
    const auto found = sm_domains.find(record.sm);
    if (found == sm_domains.end())
    {
      ++unknown;
      continue;
    }
    mismatches += found->second != static_cast<int>(block / capacity);
    intervals[found->second].push_back(record);
  }
  std::fprintf(
    stderr,
    "[overlap] device=%d reduction_blocks=%zu mismatches=%u unknown=%u\n",
    device,
    records.size(),
    mismatches,
    unknown);

  // Merge the actual intervals, since overlapping first-to-last envelopes can hide gaps.
  for (int domain = 0; domain < domain_count; ++domain)
  {
    auto& spans = intervals[domain];
    std::sort(spans.begin(), spans.end(), [](const auto& a, const auto& b) {
      return a.start < b.start;
    });
    std::vector<block_interval> merged;
    for (const auto& span : spans)
    {
      if (merged.empty() || span.start > merged.back().end)
      {
        merged.push_back(span);
      }
      else
      {
        merged.back().end = std::max(merged.back().end, span.end);
      }
    }
    unsigned long long active = 0;
    for (const auto& span : merged)
    {
      active += span.end - span.start;
    }
    if (!merged.empty())
    {
      std::fprintf(
        stderr,
        "[overlap] domain=%d blocks=%zu first_us=%.3f last_us=%.3f active_us=%.3f\n",
        domain,
        spans.size(),
        (merged.front().start - origin) / 1000.0,
        (merged.back().end - origin) / 1000.0,
        active / 1000.0);
    }
    spans = std::move(merged);
  }

  for (int a = 0; a < domain_count; ++a)
  {
    for (int b = a + 1; b < domain_count; ++b)
    {
      const auto& left           = intervals[a];
      const auto& right          = intervals[b];
      cuda::std::size_t i        = 0;
      cuda::std::size_t j        = 0;
      unsigned long long overlap = 0;
      while (i < left.size() && j < right.size())
      {
        const auto start = std::max(left[i].start, right[j].start);
        const auto end   = std::min(left[i].end, right[j].end);
        if (end > start)
        {
          overlap += end - start;
        }
        if (left[i].end < right[j].end)
        {
          ++i;
        }
        else
        {
          ++j;
        }
      }
      std::fprintf(
        stderr,
        "[overlap] domains=%d,%d concurrent_us=%.3f%s\n",
        a,
        b,
        overlap / 1000.0,
        unknown ? " INCOMPLETE: unknown SMs" : "");
    }
  }
}

_CCCL_KERNEL_ATTRIBUTES void record_block_placement(unsigned* sm_ids)
{
  if (threadIdx.x == 0)
  {
    unsigned sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    sm_ids[blockIdx.x] = sm;
  }
}

inline std::vector<unsigned> probe(CUlaunchConfig config, unsigned* device_ids)
{
  CUfunction function{};
  _CCCL_TRY_RUNTIME_API(
    cudaGetFuncBySymbol, "Resolve placement kernel", &function, reinterpret_cast<const void*>(record_block_placement));
  void* args[] = {&device_ids};
  cuda::__driver::__launchKernel(config, function, args);

  std::vector<unsigned> result(config.gridDimX);
  _CCCL_TRY_RUNTIME_API(
    cudaMemcpyAsync,
    "Read placement buffer",
    result.data(),
    device_ids,
    result.size() * sizeof(unsigned),
    cudaMemcpyDeviceToHost,
    config.hStream);
  _CCCL_TRY_RUNTIME_API(cudaStreamSynchronize, "Wait for placement probe", config.hStream);
  return result;
}

template <class Kernel, class GridMapping>
bool inspect(const CUlaunchConfig& config, Kernel kernel, const GridMapping& mapping, void** args)
{
  if (!requested)
  {
    return false;
  }
  requested = false;

  int device_id{};
  _CCCL_TRY_RUNTIME_API(cudaGetDevice, "Get placement device", &device_id);
  const auto device  = cuda::devices[device_id];
  const auto domains = device.__locality_domains();
  auto ids =
    cuda::make_device_buffer<unsigned>(cuda::stream_ref{config.hStream}, device, config.gridDimX, cuda::no_init);
  // The other streams must not access this allocation before its allocation operation completes.
  _CCCL_TRY_RUNTIME_API(cudaStreamSynchronize, "Wait for placement buffer", config.hStream);

  std::unordered_map<unsigned, int> sm_domains;
  for (int domain = 0; domain < static_cast<int>(domains.size()); ++domain)
  {
    cuda::stream local_stream{domains[domain]};
    auto local_config     = config;
    local_config.hStream  = local_stream.get();
    local_config.attrs    = nullptr;
    local_config.numAttrs = 0;
    for (unsigned sm : probe(local_config, ids.data()))
    {
      const auto [it, inserted] = sm_domains.emplace(sm, domain);
      if (!inserted && it->second != domain)
      {
        throw std::runtime_error("Placement probe observes one SM in multiple locality domains");
      }
    }
  }

  const auto observed = probe(config, ids.data());
  const int count     = static_cast<int>(domains.size());
  const int capacity  = (static_cast<int>(config.gridDimX) + count - 1) / count;
  unsigned mismatches = 0;
  unsigned unknown    = 0;
  std::fprintf(
    stderr,
    "[placement] device=%d blocks=%u threads=%u domains=%d\n",
    device_id,
    config.gridDimX,
    config.blockDimX,
    count);
  for (unsigned block = 0; block < config.gridDimX; ++block)
  {
    const int expected = static_cast<int>(block) / capacity;
    const auto found   = sm_domains.find(observed[block]);
    if (found == sm_domains.end())
    {
      ++unknown;
      std::fprintf(
        stderr, "[placement] block=%u sm=%u expected=%d observed=UNKNOWN\n", block, observed[block], expected);
    }
    else if (found->second != expected)
    {
      ++mismatches;
      std::fprintf(
        stderr,
        "[placement] block=%u sm=%u expected=%d observed=%d MISMATCH\n",
        block,
        observed[block],
        expected,
        found->second);
    }
  }
  std::fprintf(
    stderr,
    "[placement] device=%d mismatches=%u unknown=%u total=%u (probe kernel only)\n",
    device_id,
    mismatches,
    unknown,
    config.gridDimX);

  auto records = cuda::make_device_buffer<block_interval>(
    cuda::stream_ref{config.hStream}, device, config.gridDimX, block_interval{});
  timed_mapping<GridMapping> timed{mapping, records.data()};
  CUfunction function{};
  _CCCL_TRY_RUNTIME_API(
    cudaGetFuncBySymbol, "Resolve timed reduction kernel", &function, reinterpret_cast<const void*>(kernel));
  void* original_mapping = args[3];
  args[3]                = &timed;
  cuda::__driver::__launchKernel(config, function, args);
  args[3] = original_mapping;

  std::vector<block_interval> host_records(config.gridDimX);
  _CCCL_TRY_RUNTIME_API(
    cudaMemcpyAsync,
    "Read reduction intervals",
    host_records.data(),
    records.data(),
    host_records.size() * sizeof(block_interval),
    cudaMemcpyDeviceToHost,
    config.hStream);
  _CCCL_TRY_RUNTIME_API(cudaStreamSynchronize, "Wait for timed reduction", config.hStream);
  report_overlap(host_records, sm_domains, device_id, count);
  return true;
}
} // namespace striped_reduce_placement

#define CCCL_STRIPED_REDUCE_PLACEMENT_PROBE(config, kernel, mapping, args) \
  ::striped_reduce_placement::inspect(config, kernel, mapping, args)
// END TEMPORARY STRIPED PLACEMENT DIAGNOSTICS

#include <cub/device/device_reduce.cuh>

#include <cuda/__device/logical_device_ref.h>
#include <cuda/__driver/driver_api.h>
#include <cuda/__event/event.h>
#include <cuda/__memory_pool/locality_domain_memory_pool.h>
#include <cuda/__runtime/api_wrapper.h>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/ranges>
#include <cuda/std/span>
#include <cuda/stream>

#include <cuda/experimental/memory_resource.cuh>
#include <cuda/experimental/mgmn/nccl_communicators>
#include <cuda/experimental/mgmn/reduce>

#include <stdexcept>
#include <string>
#include <vector>

#include <nccl.h>

#include <nvbench/nvbench.cuh>

namespace cudax = cuda::experimental;

using T = float;

namespace
{
[[nodiscard]] std::vector<cudax::mgmn::nccl_communicator>
make_communicators(cuda::device_ref device, cuda::std::span<const cuda::__logical_device_ref> domains)
{
  std::vector<ncclComm_t> raw_comms(domains.size());
  std::vector<int> devs(domains.size(), device.get());
  std::vector<cudax::mgmn::nccl_communicator> comms;
  comms.reserve(domains.size());

  if (const auto status = ncclCommInitAll(raw_comms.data(), static_cast<int>(devs.size()), devs.data());
      status != ncclSuccess)
  {
    throw std::runtime_error(std::string{"ncclCommInitAll: "} + ncclGetErrorString(status));
  }

  for (cuda::std::size_t domain = 0; domain < domains.size(); ++domain)
  {
    comms.push_back(cudax::mgmn::nccl_communicator::from_native_handle(raw_comms[domain], domains[domain]));
  }
  return comms;
}

template <typename SubmitFn>
void run_forked_iteration(
  cuda::stream_ref timing_stream,
  std::vector<cuda::stream>& streams,
  cuda::event& fork,
  std::vector<cuda::event>& join,
  SubmitFn submit)
{
  fork.record(timing_stream);
  for (auto& stream : streams)
  {
    stream.wait(fork);
  }

  submit();

  // All records before any waits: interleaving them blocks the host between records.
  for (auto&& [event, stream] : cuda::std::views::zip(join, streams))
  {
    event.record(stream);
  }
  for (auto& event : join)
  {
    timing_stream.wait(event);
  }
}

void cudax_reduce(nvbench::state& state)
{
  const auto elements = static_cast<cuda::std::size_t>(state.get_int64("Elements"));
  const auto device  = cuda::devices[state.get_device().value().get_id()]; // NOLINT(bugprone-unchecked-optional-access)
  const auto domains = device.__locality_domains();
  const auto rank_count = domains.size();

  if (rank_count < 2)
  {
    state.skip("the GPU does not expose multiple locality domains");
    return;
  }

  if (elements % rank_count != 0)
  {
    throw std::runtime_error{"Elements must divide evenly across locality domains"};
  }

  const auto per_rank = elements / rank_count;

  auto comms = make_communicators(device, domains);
  std::vector<cuda::stream> streams{domains.begin(), domains.end()};
  std::vector<cuda::device_buffer<T>> inputs;
  std::vector<cuda::device_buffer<T>> outputs;
  cuda::event fork{device};
  std::vector<cuda::event> join;

  inputs.reserve(rank_count);
  outputs.reserve(rank_count);
  join.reserve(rank_count);
  for (cuda::std::size_t rank = 0; rank < rank_count; ++rank)
  {
    auto& pool = cuda::__device_default_memory_pool(domains[rank]);

    inputs.push_back(cuda::make_buffer<T>(streams[rank], pool, per_rank, T{1}));
    outputs.push_back(cuda::make_buffer<T>(streams[rank], pool, 1, cuda::no_init));
    join.emplace_back(device);
  }

  for (auto& stream : streams)
  {
    stream.sync();
  }

  // Reject non-localized allocations before measuring locality-sensitive bandwidth.
  for (cuda::std::size_t rank = 0; rank < rank_count; ++rank)
  {
    if (inputs[rank].empty())
    {
      continue;
    }

    unsigned int ordinal        = ~0U;
    CUpointer_attribute attrs[] = {CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL};
    void* values[]              = {&ordinal};

    _CCCL_TRY_DRIVER_API(
      cuda::__driver::__pointerGetAttributesNoThrow,
      "Failed to query the locality domain of a pointer",
      attrs,
      values,
      inputs[rank].data());

    if (ordinal != domains[rank].locality_domain().domain_id)
    {
      state.skip("an input buffer did not land in its requested locality domain");
      return;
    }
  }

  // Use the localized pool for the reduction's temporary storage as well as its input.
  auto environments = inputs | cuda::std::views::transform([](auto& input) {
                        return cuda::std::execution::env{
                          input.stream(),
                          input.memory_resource(),
                          cuda::execution::require(cuda::execution::determinism::not_guaranteed)};
                      });

  state.add_element_count(elements, "Elements");
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(rank_count);

  auto& summary = state.add_summary("mgmn/locality_domains");

  summary.set_string("name", "Domains");
  summary.set_string("hint", "");
  summary.set_string("description", "Locality domains the device was split into");
  summary.set_int64("value", static_cast<cuda::std::int64_t>(rank_count));

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    run_forked_iteration(cuda::stream_ref{launch.get_stream().get_stream()}, streams, fork, join, [&] {
      cudax::mgmn::reduce(
        cudax::broadcasted,
        comms,
        environments,
        inputs | cuda::std::views::transform(cuda::std::ranges::data),
        inputs | cuda::std::views::transform(cuda::std::ranges::size),
        outputs | cuda::std::views::transform(cuda::std::ranges::data));
    });
  });
}

constexpr int min_elements_pow2 = 28;
constexpr int max_elements_pow2 = 32;
constexpr int elements_stride   = 1;
} // namespace

NVBENCH_BENCH(cudax_reduce)
  .set_name("cudax_reduce")
  .add_int64_power_of_two_axis("Elements", nvbench::range(min_elements_pow2, max_elements_pow2, elements_stride));

namespace
{
void cub_reduce_striped(nvbench::state& state)
{
  const auto elements = static_cast<cuda::std::size_t>(state.get_int64("Elements"));
  const auto device  = cuda::devices[state.get_device().value().get_id()]; // NOLINT(bugprone-unchecked-optional-access)
  const auto domains = device.__locality_domains();
  const auto rank_count = domains.size();

  if (rank_count < 2)
  {
    state.skip("the GPU does not expose multiple locality domains");
    return;
  }
  if (elements % rank_count != 0)
  {
    throw std::runtime_error{"Elements must divide evenly across locality domains"};
  }
  const auto per_rank = elements / rank_count;

  cuda::stream stream{device};
  cudax::locality_domain_striped_memory_resource resource{device, per_rank * sizeof(T)};
  const auto input = cuda::make_buffer<T>(stream, resource, elements, T{1});
  auto output      = cuda::make_device_buffer<T>(stream, device, 1, cuda::no_init);
  stream.sync();

  state.add_element_count(elements, "Elements");
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(1);

  auto& summary = state.add_summary("mgmn/locality_domains");
  summary.set_string("name", "Domains");
  summary.set_string("hint", "");
  summary.set_string("description", "Locality domains the device was split into");
  summary.set_int64("value", static_cast<cuda::std::int64_t>(rank_count));

  // BEGIN TEMPORARY STRIPED PLACEMENT DIAGNOSTICS
  // An untimed reduction supplies the exact grid and launch attribute to the probe.
  {
    bool memory_matches = true;
    for (cuda::std::size_t rank = 0; per_rank != 0 && rank < rank_count; ++rank)
    {
      const auto* stripe      = reinterpret_cast<const char*>(input.data() + rank * per_rank);
      const void* endpoints[] = {stripe, stripe + per_rank * sizeof(T) - 1};
      unsigned ordinals[]     = {~0U, ~0U};
      for (int endpoint = 0; endpoint < 2; ++endpoint)
      {
        CUpointer_attribute attrs[] = {CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL};
        void* values[]              = {&ordinals[endpoint]};
        _CCCL_TRY_DRIVER_API(
          cuda::__driver::__pointerGetAttributesNoThrow,
          "Failed to query the locality domain of a stripe endpoint",
          attrs,
          values,
          endpoints[endpoint]);
      }

      const auto expected = static_cast<unsigned>(domains[rank].locality_domain().domain_id);
      const bool matches  = ordinals[0] == expected && ordinals[1] == expected;
      memory_matches      = memory_matches && matches;
      std::fprintf(
        stderr,
        "[memory-placement] device=%d stripe=%zu expected=%u first=%u last=%u %s\n",
        device.get(),
        rank,
        expected,
        ordinals[0],
        ordinals[1],
        matches ? "MATCH" : "MISMATCH");
    }
    if (!memory_matches)
    {
      throw std::runtime_error("Striped input memory does not match the requested locality domains");
    }

    const auto env = cuda::std::execution::env{
      cuda::stream_ref{stream.get()},
      input.memory_resource(),
      cuda::execution::require(cuda::execution::determinism::not_guaranteed),
      cub::striped_marker_t{static_cast<int>(rank_count)}};
    striped_reduce_placement::requested = true;
    _CCCL_TRY_RUNTIME_API(
      cub::DeviceReduce::Reduce,
      "Placement diagnostic reduction failed",
      input.data(),
      output.data(),
      input.size(),
      cuda::std::plus<>{},
      T{},
      env);
    stream.sync();
    if (striped_reduce_placement::requested)
    {
      striped_reduce_placement::requested = false;
      throw std::runtime_error("Striped reduction did not reach the placement probe");
    }
  }
  // END TEMPORARY STRIPED PLACEMENT DIAGNOSTICS

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    const auto env = cuda::std::execution::env{
      cuda::stream_ref{launch.get_stream().get_stream()},
      cuda::execution::require(cuda::execution::determinism::not_guaranteed),
      cub::striped_marker_t{static_cast<int>(rank_count)}};

    _CCCL_TRY_RUNTIME_API(
      cub::DeviceReduce::Reduce,
      "cub::DeviceReduce::Reduce failed",
      input.data(),
      output.data(),
      input.size(),
      cuda::std::plus<>{},
      T{},
      env);
  });
}
} // namespace

NVBENCH_BENCH(cub_reduce_striped)
  .set_name("cub_reduce_striped")
  .add_int64_power_of_two_axis("Elements", nvbench::range(min_elements_pow2, max_elements_pow2, elements_stride));
