#!/usr/bin/env python3
"""Run and summarize the independently built MGMN reduction benchmarks."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import statistics
import subprocess
from typing import Any

SCENARIOS = (
    "cub_full_device",
    "cudax_host_nccl",
    "cub_green_atomic",
    "cudax_device_nccl",
)


def parse_sizes(value: str) -> list[int]:
    try:
        sizes = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "sizes must be a comma-separated list of integers"
        ) from error
    if not sizes or any(size <= 0 for size in sizes) or len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("sizes must be positive and unique")
    return sizes


def make_sizes(args: argparse.Namespace) -> list[int]:
    if args.sizes:
        sizes = args.sizes
    else:
        if args.range_multiplier < 2:
            raise ValueError("--range-multiplier must be at least two")
        if args.min_elements <= 0 or args.max_elements < args.min_elements:
            raise ValueError("invalid element range")
        sizes = []
        size = args.min_elements
        while size <= args.max_elements:
            sizes.append(size)
            if size > args.max_elements // args.range_multiplier:
                break
            size *= args.range_multiplier
        if (
            sizes[-1] != args.max_elements
            and sizes[-1] * args.range_multiplier <= args.max_elements
        ):
            raise ValueError("element range does not terminate safely")
    if any(size % 2 for size in sizes):
        raise ValueError(
            "all sizes must be divisible by two for green-context scenarios"
        )
    return sizes


def run(
    command: list[str],
    log: pathlib.Path,
    verbose: bool,
    dry_run: bool,
    env: dict[str, str] | None = None,
) -> int:
    if verbose or dry_run:
        print("+", " ".join(command))
    if dry_run:
        return 0
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            output.write(line)
            if verbose:
                print(line, end="")
        return process.wait()


def summarize(results: dict[str, Any], log_dir: pathlib.Path) -> dict[str, Any]:
    rows: dict[int, dict[str, Any]] = {}
    for scenario, result in results.items():
        if result.get("status") != "ok":
            continue
        for benchmark in result["json"].get("benchmarks", []):
            elements = benchmark.get("elements")
            if elements is None or benchmark.get("run_type") == "aggregate":
                continue
            row = rows.setdefault(
                int(elements),
                {"elements": int(elements), "input_bytes": int(elements) * 4},
            )
            row.setdefault(scenario, []).append(float(benchmark["real_time"]) / 1e9)
    summary_rows = []
    for elements in sorted(rows):
        row = rows[elements]
        for scenario in SCENARIOS:
            samples = row.get(scenario, [])
            if not samples:
                row[scenario] = {
                    "status": results[scenario]["status"],
                    "median_seconds": None,
                }
                continue
            median = statistics.median(samples)
            mean = statistics.fmean(samples)
            deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
            row[scenario] = {
                "status": "ok",
                "iterations": len(samples),
                "median_seconds": median,
                "mean_seconds": mean,
                "stddev_seconds": deviation,
                "coefficient_of_variation": deviation / mean if mean else 0.0,
                "gb_per_second": row["input_bytes"] / median / 1e9,
                "raw_seconds": samples,
            }
        baseline = row["cub_full_device"]["median_seconds"]
        for scenario in SCENARIOS:
            measurement = row[scenario]
            measurement["latency_speedup"] = (
                baseline / measurement["median_seconds"]
                if baseline and measurement["median_seconds"]
                else None
            )
            measurement["throughput_ratio"] = (
                measurement["gb_per_second"] / row["cub_full_device"]["gb_per_second"]
                if baseline and measurement["median_seconds"]
                else None
            )
        summary_rows.append(row)
    return {"log_dir": str(log_dir), "scenarios": results, "rows": summary_rows}


def write_markdown(summary: dict[str, Any], path: pathlib.Path) -> None:
    lines = [
        "# MGMN benchmark summary",
        "",
        "| Elements | Scenario | Median ms | GB/s | Speedup | Status |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        for scenario in SCENARIOS:
            value = row[scenario]
            if value["median_seconds"] is None:
                lines.append(
                    f"| {row['elements']} | {scenario} | N/A | N/A | N/A | {value['status']} |"
                )
            else:
                lines.append(
                    f"| {row['elements']} | {scenario} | {value['median_seconds'] * 1e3:.3f} | {value['gb_per_second']:.3f} | {value['latency_speedup']:.3f} | ok |"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", action="count", default=0)
    parser.add_argument(
        "--binary-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent.parent.parent
        / "build"
        / "mgmn_bench"
        / "benchmarks"
        / "mgmn",
    )
    parser.add_argument("--log-dir", type=pathlib.Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--min-elements", type=int, default=1 << 10)
    parser.add_argument("--max-elements", type=int, default=1 << 28)
    parser.add_argument("--range-multiplier", type=int, default=4)
    parser.add_argument("--sizes", type=parse_sizes)
    parser.add_argument("--min-time", type=float, default=0.5)
    parser.add_argument("--warmup-time", type=float, default=0.1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--benchmark-filter", default=".*")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sizes = make_sizes(args)
    root = pathlib.Path(__file__).resolve().parent
    log_dir = args.log_dir or root / "benchmark-results" / dt.datetime.now(
        dt.timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "sizes": sizes,
        "device": args.device,
        "scenarios": list(SCENARIOS),
        "commands": [],
    }
    (log_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    results: dict[str, Any] = {}
    for scenario in SCENARIOS:
        json_path = log_dir / f"{scenario}.json"
        executable = args.binary_dir / scenario
        if not args.dry_run and not executable.is_file():
            raise FileNotFoundError(f"missing benchmark executable: {executable}")
        command = [
            str(executable),
            f"--cccl-benchmark-sizes={','.join(map(str, sizes))}",
            f"--benchmark_min_time={args.min_time}s",
            f"--benchmark_min_warmup_time={args.warmup_time}",
            f"--benchmark_repetitions={args.repetitions}",
            "--benchmark_display_aggregates_only=true",
            "--benchmark_report_aggregates_only=false",
            "--benchmark_out_format=json",
            f"--benchmark_out={json_path}",
            f"--benchmark_filter={args.benchmark_filter}",
        ]
        manifest["commands"].append(command)
        env = os.environ | {"CUDA_VISIBLE_DEVICES": str(args.device)}
        if "nccl" in scenario:
            env |= {
                "NCCL_MULTI_RANK_GPU_ENABLE": "1",
                "NCCL_NVLS_ENABLE": "0",
                "NCCL_MAX_CTAS": "1",
                "NCCL_DEBUG": "INFO" if args.v > 1 else "WARN",
            }
        status = run(
            command, log_dir / f"{scenario}.log", args.v > 0, args.dry_run, env
        )
        results[scenario] = {
            "status": "ok" if status == 0 else f"failed ({status})",
            "log": str(log_dir / f"{scenario}.log"),
        }
        if status == 0 and not args.dry_run:
            results[scenario]["json"] = json.loads(
                json_path.read_text(encoding="utf-8")
            )
    (log_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    summary = (
        summarize(results, log_dir)
        if not args.dry_run
        else {"log_dir": str(log_dir), "scenarios": results, "rows": []}
    )
    (log_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(summary, log_dir / "summary.md")
    print(log_dir / "summary.md")
    return 0 if all(value["status"] == "ok" for value in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
