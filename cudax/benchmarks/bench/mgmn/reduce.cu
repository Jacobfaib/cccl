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
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace striped_reduce_placement
{
inline thread_local bool requested = false;

template <class Kernel, class Policy>
void describe_kernel(
  Kernel kernel, Policy policy, cudaStream_t stream, int grid, int sms, bool striped, int offset_bytes)
{
  int device{};
  _CCCL_TRY_RUNTIME_API(cudaGetDevice, "Get kernel device", &device);
  const auto pointer = reinterpret_cast<const void*>(kernel);
  using key_type     = std::tuple<int, const void*, cudaStream_t, int>;
  static std::vector<key_type> reported;
  const key_type key{device, pointer, stream, grid};
  if (std::find(reported.begin(), reported.end(), key) != reported.end())
  {
    return;
  }
  reported.push_back(key);
  cudaFuncAttributes attributes{};
  int resident{};
  _CCCL_TRY_RUNTIME_API(cudaFuncGetAttributes, "Get reduction attributes", &attributes, pointer);
  _CCCL_TRY_RUNTIME_API(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor,
    "Get reduction occupancy",
    &resident,
    pointer,
    policy.threads_per_block,
    0);
  std::fprintf(
    stderr,
    "[kernel] device=%d striped=%d stream=%p grid=%d sms=%d blocks_per_sm=%d waves=%.2f "
    "threads=%d items=%d vector=%d algorithm=%d load=%d offset_bytes=%d "
    "registers=%d shared_bytes=%zu local_bytes=%zu binary=%d ptx=%d\n",
    device,
    striped,
    reinterpret_cast<void*>(stream),
    grid,
    sms,
    resident,
    static_cast<double>(grid) / (sms * resident),
    policy.threads_per_block,
    policy.items_per_thread,
    policy.vec_size,
    static_cast<int>(policy.reduce_algorithm),
    static_cast<int>(policy.load_modifier),
    offset_bytes,
    attributes.numRegs,
    attributes.sharedSizeBytes,
    attributes.localSizeBytes,
    attributes.binaryVersion,
    attributes.ptxVersion);
}

struct block_interval
{
  unsigned long long start;
  unsigned long long end;
  unsigned sm;
  int threads;
  int items;
  int vector;
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

  _CCCL_DEVICE _CCCL_FORCEINLINE void record_end(int threads, int items, int vector)
  {
    __syncthreads();
    if (threadIdx.x == 0)
    {
      unsigned long long end_time;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end_time) : : "memory");
      records[blockIdx.x] = {start_time, end_time, sm, threads, items, vector};
    }
  }
};

template <class GridMapping>
inline constexpr bool is_timed_mapping = false;

template <class GridMapping>
inline constexpr bool is_timed_mapping<timed_mapping<GridMapping>> = true;

struct thread_load_range
{
  unsigned long long begin = ~0ULL;
  unsigned long long end{};
  unsigned long long expected_begin{};
  unsigned long long expected_end{};
  unsigned long long bytes{};
  unsigned long long violations{};
  unsigned sm{};
  int domain = -1;
};

template <class GridMapping>
struct access_mapping : GridMapping
{
  thread_load_range* output;
  const int* sm_domains;
  unsigned sm_count;
  unsigned long long input_begin;
  unsigned long long stripe_bytes;
  thread_load_range range{};

  _CCCL_DEVICE _CCCL_FORCEINLINE void record_start()
  {
    asm volatile("mov.u32 %0, %%smid;" : "=r"(range.sm));
    range.domain = range.sm < sm_count ? sm_domains[range.sm] : -1;
    if (range.domain >= 0)
    {
      range.expected_begin = input_begin + range.domain * stripe_bytes;
      range.expected_end   = range.expected_begin + stripe_bytes;
    }
  }

  template <class Pointer>
  _CCCL_DEVICE _CCCL_FORCEINLINE void record_load(Pointer pointer, unsigned bytes)
  {
    const auto begin = reinterpret_cast<unsigned long long>(pointer);
    const auto end   = begin + bytes;
    range.begin      = begin < range.begin ? begin : range.begin;
    range.end        = end > range.end ? end : range.end;
    range.bytes += bytes;
    range.violations += range.domain < 0 || begin < range.expected_begin || end > range.expected_end || end < begin;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void record_end(int, int, int)
  {
    output[static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x] = range;
  }
};

template <class GridMapping>
inline constexpr bool is_timed_mapping<access_mapping<GridMapping>> = true;

template <class GridMapping>
inline constexpr bool is_access_mapping = false;

template <class GridMapping>
inline constexpr bool is_access_mapping<access_mapping<GridMapping>> = true;

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
    std::vector<unsigned> used_sms;
    std::vector<std::pair<unsigned long long, int>> edges;
    for (const auto& span : spans)
    {
      used_sms.push_back(span.sm);
      edges.emplace_back(span.start, 1);
      edges.emplace_back(span.end, -1);
    }
    std::sort(used_sms.begin(), used_sms.end());
    used_sms.erase(std::unique(used_sms.begin(), used_sms.end()), used_sms.end());
    std::sort(edges.begin(), edges.end());
    int live = 0;
    int peak = 0;
    for (const auto& edge : edges)
    {
      live += edge.second;
      peak = std::max(peak, live);
    }
    std::fprintf(stderr, "[residency] domain=%d sms_used=%zu peak_live_blocks=%d\n", domain, used_sms.size(), peak);
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

template <class Kernel, class NormalKernel, class AccessKernel, class GridMapping, class Policy>
bool inspect(
  CUlaunchConfig config,
  Kernel kernel,
  NormalKernel normal_kernel,
  AccessKernel access_kernel,
  const GridMapping& mapping,
  void** args,
  Policy policy,
  int sms,
  int output_bytes)
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

  const auto normal_pointer     = reinterpret_cast<const void*>(normal_kernel);
  const auto diagnostic_pointer = reinterpret_cast<const void*>(kernel);
  cudaFuncAttributes normal_attributes{};
  cudaFuncAttributes diagnostic_attributes{};
  int normal_resident{};
  int diagnostic_resident{};
  _CCCL_TRY_RUNTIME_API(cudaFuncGetAttributes, "Get normal attributes", &normal_attributes, normal_pointer);
  _CCCL_TRY_RUNTIME_API(cudaFuncGetAttributes, "Get diagnostic attributes", &diagnostic_attributes, diagnostic_pointer);
  _CCCL_TRY_RUNTIME_API(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor,
    "Get normal occupancy",
    &normal_resident,
    normal_pointer,
    config.blockDimX,
    0);
  _CCCL_TRY_RUNTIME_API(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor,
    "Get diagnostic occupancy",
    &diagnostic_resident,
    diagnostic_pointer,
    config.blockDimX,
    0);
  std::fprintf(
    stderr,
    "[occupancy] normal_regs=%d diagnostic_regs=%d normal_shared=%zu diagnostic_shared=%zu "
    "normal_blocks_per_sm=%d diagnostic_blocks_per_sm=%d sms=%d\n",
    normal_attributes.numRegs,
    diagnostic_attributes.numRegs,
    normal_attributes.sharedSizeBytes,
    diagnostic_attributes.sharedSizeBytes,
    normal_resident,
    diagnostic_resident,
    sms);
  if (normal_resident == 0 || diagnostic_resident == 0)
  {
    throw std::runtime_error("Reduction diagnostic has zero theoretical occupancy");
  }

  const int resident_grid = sms * std::min(normal_resident, diagnostic_resident);
  std::vector<int> grids{resident_grid / 2, resident_grid, resident_grid * 2, static_cast<int>(config.gridDimX)};
  std::sort(grids.begin(), grids.end());
  grids.erase(std::unique(grids.begin(), grids.end()), grids.end());

  CUfunction function{};
  CUfunction normal_function{};
  _CCCL_TRY_RUNTIME_API(
    cudaGetFuncBySymbol, "Resolve timed reduction kernel", &function, reinterpret_cast<const void*>(kernel));
  _CCCL_TRY_RUNTIME_API(cudaGetFuncBySymbol, "Resolve normal reduction kernel", &normal_function, normal_pointer);
  // This benchmark uses pointer outputs. Clear the atomic destination before every diagnostic launch.
  void* output = *static_cast<void**>(args[1]);
  if (output_bytes != sizeof(float))
  {
    throw std::runtime_error("Grid sweep expects the benchmark's float output");
  }
  const float expected_sum = static_cast<float>(mapping.num_items);
  cudaEvent_t start{};
  cudaEvent_t stop{};
  _CCCL_TRY_RUNTIME_API(cudaEventCreate, "Create diagnostic start event", &start);
  _CCCL_TRY_RUNTIME_API(cudaEventCreate, "Create diagnostic stop event", &stop);
  void* original_mapping = args[3];
  for (int grid : grids)
  {
    auto sweep_mapping = mapping;
    sweep_mapping.DispatchInit(
      mapping.num_items, std::max(count, grid), policy.threads_per_block * policy.items_per_thread);
    auto sweep_config     = config;
    sweep_config.gridDimX = sweep_mapping.grid_size;
    args[3]               = &sweep_mapping;
    float elapsed_sum     = 0;
    for (int repeat = 0; repeat < 4; ++repeat)
    {
      _CCCL_TRY_RUNTIME_API(cudaMemsetAsync, "Reset diagnostic output", output, 0, output_bytes, config.hStream);
      _CCCL_TRY_RUNTIME_API(cudaEventRecord, "Record reduction start", start, config.hStream);
      cuda::__driver::__launchKernel(sweep_config, normal_function, args);
      _CCCL_TRY_RUNTIME_API(cudaEventRecord, "Record reduction stop", stop, config.hStream);
      _CCCL_TRY_RUNTIME_API(cudaEventSynchronize, "Wait for reduction stop", stop);
      float elapsed{};
      _CCCL_TRY_RUNTIME_API(cudaEventElapsedTime, "Measure reduction", &elapsed, start, stop);
      if (repeat != 0)
      {
        elapsed_sum += elapsed;
      }
    }
    std::fprintf(
      stderr,
      "[grid-sweep] grid=%u normal_mean_us=%.3f threads=%u items=%d vector=%d\n",
      sweep_config.gridDimX,
      elapsed_sum * 1000.0 / 3,
      sweep_config.blockDimX,
      policy.items_per_thread,
      policy.vec_size);

    float result{};
    _CCCL_TRY_RUNTIME_API(
      cudaMemcpy, "Check grid sweep result", &result, output, sizeof(result), cudaMemcpyDeviceToHost);
    if (result != expected_sum)
    {
      throw std::runtime_error("Grid sweep reduction does not match the all-ones input sum");
    }

    auto records = cuda::make_device_buffer<block_interval>(
      cuda::stream_ref{config.hStream}, device, sweep_config.gridDimX, block_interval{});
    timed_mapping<GridMapping> timed{sweep_mapping, records.data()};
    args[3] = &timed;
    _CCCL_TRY_RUNTIME_API(cudaMemsetAsync, "Reset timed diagnostic output", output, 0, output_bytes, config.hStream);
    _CCCL_TRY_RUNTIME_API(cudaEventRecord, "Record diagnostic start", start, config.hStream);
    cuda::__driver::__launchKernel(sweep_config, function, args);
    _CCCL_TRY_RUNTIME_API(cudaEventRecord, "Record diagnostic stop", stop, config.hStream);
    std::vector<block_interval> host_records(sweep_config.gridDimX);
    _CCCL_TRY_RUNTIME_API(
      cudaMemcpyAsync,
      "Read reduction intervals",
      host_records.data(),
      records.data(),
      host_records.size() * sizeof(block_interval),
      cudaMemcpyDeviceToHost,
      config.hStream);
    _CCCL_TRY_RUNTIME_API(cudaStreamSynchronize, "Wait for timed reduction", config.hStream);
    float elapsed{};
    _CCCL_TRY_RUNTIME_API(cudaEventElapsedTime, "Measure diagnostic reduction", &elapsed, start, stop);
    std::fprintf(stderr, "[grid-sweep] grid=%u instrumented_us=%.3f\n", sweep_config.gridDimX, elapsed * 1000.0);
    const auto& device_policy = host_records.front();
    std::fprintf(stderr,
                 "[device-policy] threads=%d items=%d vector=%d\n",
                 device_policy.threads,
                 device_policy.items,
                 device_policy.vector);
    if (device_policy.threads != policy.threads_per_block || device_policy.items != policy.items_per_thread
        || device_policy.vector != policy.vec_size)
    {
      throw std::runtime_error("Host and device reduction policies do not match");
    }
    report_overlap(host_records, sm_domains, device_id, count);
    _CCCL_TRY_RUNTIME_API(
      cudaMemcpy, "Check instrumented result", &result, output, sizeof(result), cudaMemcpyDeviceToHost);
    if (result != expected_sum)
    {
      throw std::runtime_error("Instrumented reduction does not match the all-ones input sum");
    }
    std::fprintf(
      stderr, "[grid-sweep] grid=%u result=%g expected=%g MATCH\n", sweep_config.gridDimX, result, expected_sum);
  }
  args[3] = original_mapping;
  _CCCL_TRY_RUNTIME_API(cudaEventDestroy, "Destroy diagnostic start event", start);
  _CCCL_TRY_RUNTIME_API(cudaEventDestroy, "Destroy diagnostic stop event", stop);

  unsigned sm_table_size = 0;
  for (const auto& entry : sm_domains)
  {
    sm_table_size = std::max(sm_table_size, entry.first + 1);
  }
  std::vector<int> sm_table(sm_table_size, -1);
  for (const auto& entry : sm_domains)
  {
    sm_table[entry.first] = entry.second;
  }
  auto device_sm_table =
    cuda::make_device_buffer<int>(cuda::stream_ref{config.hStream}, device, sm_table_size, cuda::no_init);
  _CCCL_TRY_RUNTIME_API(
    cudaMemcpyAsync,
    "Copy SM domain map",
    device_sm_table.data(),
    sm_table.data(),
    sm_table.size() * sizeof(int),
    cudaMemcpyHostToDevice,
    config.hStream);
  const auto thread_count = static_cast<cuda::std::size_t>(config.gridDimX) * config.blockDimX;
  auto addresses          = cuda::make_device_buffer<thread_load_range>(
    cuda::stream_ref{config.hStream}, device, thread_count, thread_load_range{});
  const auto input_begin  = reinterpret_cast<unsigned long long>(*static_cast<const void**>(args[0]));
  const auto stripe_bytes = static_cast<unsigned long long>(mapping.num_items / count) * sizeof(float);
  access_mapping<GridMapping> access{
    mapping, addresses.data(), device_sm_table.data(), sm_table_size, input_begin, stripe_bytes};
  CUfunction access_function{};
  _CCCL_TRY_RUNTIME_API(
    cudaGetFuncBySymbol, "Resolve address audit kernel", &access_function, reinterpret_cast<const void*>(access_kernel));
  args[3] = &access;
  _CCCL_TRY_RUNTIME_API(cudaMemsetAsync, "Reset address audit output", output, 0, output_bytes, config.hStream);
  cuda::__driver::__launchKernel(config, access_function, args);
  args[3] = original_mapping;
  std::vector<thread_load_range> host_addresses(thread_count);
  _CCCL_TRY_RUNTIME_API(
    cudaMemcpyAsync,
    "Read device load addresses",
    host_addresses.data(),
    addresses.data(),
    thread_count * sizeof(thread_load_range),
    cudaMemcpyDeviceToHost,
    config.hStream);
  _CCCL_TRY_RUNTIME_API(cudaStreamSynchronize, "Wait for address audit", config.hStream);
  unsigned long long unknown_threads = 0;
  for (const auto& range : host_addresses)
  {
    unknown_threads += range.domain < 0;
  }
  std::fprintf(stderr,
               "[device-access] checked_threads=%zu unknown_threads=%llu end_addresses=exclusive\n",
               thread_count,
               unknown_threads);
  for (int domain = 0; domain < count; ++domain)
  {
    unsigned long long begin       = ~0ULL;
    unsigned long long end         = 0;
    unsigned long long bytes       = 0;
    unsigned long long violations  = 0;
    cuda::std::size_t first_thread = 0;
    cuda::std::size_t last_thread  = 0;
    for (cuda::std::size_t thread = 0; thread < host_addresses.size(); ++thread)
    {
      const auto& range = host_addresses[thread];
      if (range.domain != domain || range.bytes == 0)
      {
        continue;
      }
      if (range.begin < begin)
      {
        begin        = range.begin;
        first_thread = thread;
      }
      if (range.end > end)
      {
        end         = range.end;
        last_thread = thread;
      }
      bytes += range.bytes;
      violations += range.violations;
    }
    const auto expected_begin = input_begin + domain * stripe_bytes;
    const auto expected_end   = expected_begin + stripe_bytes;
    const bool matches = begin == expected_begin && end == expected_end && bytes == stripe_bytes && violations == 0;
    std::fprintf(
      stderr,
      "[device-access] domain=%d first_block=%zu first_thread=%zu load_begin=0x%llx "
      "last_block=%zu last_thread=%zu load_end=0x%llx expected_begin=0x%llx expected_end=0x%llx "
      "bytes=%llu expected_bytes=%llu violating_loads=%llu %s\n",
      domain,
      first_thread / config.blockDimX,
      first_thread % config.blockDimX,
      begin,
      last_thread / config.blockDimX,
      last_thread % config.blockDimX,
      end,
      expected_begin,
      expected_end,
      bytes,
      stripe_bytes,
      violations,
      matches && unknown_threads == 0 ? "MATCH" : "MISMATCH");
  }
  return true;
}
} // namespace striped_reduce_placement

#define CCCL_STRIPED_REDUCE_PLACEMENT_PROBE(config, kernel, normal, access, mapping, args, policy, sms, output_bytes) \
  ::striped_reduce_placement::inspect(config, kernel, normal, access, mapping, args, policy, sms, output_bytes)
#define CCCL_REDUCE_KERNEL_DIAGNOSTICS(kernel, policy, stream, grid, sms, striped, offset_bytes) \
  ::striped_reduce_placement::describe_kernel(kernel, policy, stream, grid, sms, striped, offset_bytes)
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
