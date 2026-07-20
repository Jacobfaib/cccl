// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <benchmark/benchmark.h>

namespace mgmn
{
inline std::size_t parse_size(std::string_view value)
{
  std::size_t result{};
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size() || result == 0)
  {
    throw std::invalid_argument("invalid positive element count: " + std::string{value});
  }
  return result;
}

inline std::vector<std::size_t> parse_sizes(std::string_view values)
{
  std::vector<std::size_t> result;
  while (!values.empty())
  {
    const auto separator   = values.find(',');
    const std::size_t size = parse_size(values.substr(0, separator));
    if (size > static_cast<std::size_t>((std::numeric_limits<int>::max)())
        || std::find(result.begin(), result.end(), size) != result.end())
    {
      throw std::invalid_argument("element counts must be unique and representable by int");
    }
    result.push_back(size);
    values.remove_prefix(separator == std::string_view::npos ? values.size() : separator + 1);
  }
  return result;
}

inline std::vector<std::size_t> take_sizes_option(int& argc, char** argv)
{
  constexpr std::string_view prefix = "--cccl-benchmark-sizes=";
  std::vector<std::size_t> sizes;
  int output = 1;
  for (int input = 1; input < argc; ++input)
  {
    const std::string_view arg{argv[input]};
    if (arg.starts_with(prefix))
    {
      if (!sizes.empty())
      {
        throw std::invalid_argument("--cccl-benchmark-sizes may only be specified once");
      }
      sizes = parse_sizes(arg.substr(prefix.size()));
    }
    else
    {
      argv[output++] = argv[input];
    }
  }
  argc = output;
  if (sizes.empty())
  {
    throw std::invalid_argument("missing --cccl-benchmark-sizes option");
  }
  return sizes;
}

inline void set_common_counters(benchmark::State& state, std::size_t elements, unsigned int sm_count = 0)
{
  state.counters["elements"]    = static_cast<double>(elements);
  state.counters["input_bytes"] = static_cast<double>(elements * sizeof(float));
  if (sm_count != 0)
  {
    state.counters["sm_count"] = static_cast<double>(sm_count);
  }
}

template <typename BenchmarkT>
int run_benchmark(int argc, char** argv, const char* scenario, BenchmarkT benchmark_function)
{
  const auto sizes = take_sizes_option(argc, argv);
  benchmark::Initialize(&argc, argv);
  benchmark::AddCustomContext("scenario", scenario);
  auto* registration = benchmark::RegisterBenchmark(std::string{scenario} + "/elements", benchmark_function);
  registration->UseManualTime();
  for (const auto size : sizes)
  {
    registration->Arg(static_cast<std::int64_t>(size));
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
} // namespace mgmn
