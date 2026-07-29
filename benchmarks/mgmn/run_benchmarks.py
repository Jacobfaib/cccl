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
#    "cudax_device_nccl",
)

BASELINE = SCENARIOS[0]

#! Peak memory bandwidth per CUDA device, in GB/s, keyed by (compute capability, total
#! memory in MiB as reported by nvidia-smi).
#!
#! `nvidia-smi` exposes no bus width on any current driver branch, so the usual
#! `2 * clock * width / 8` derivation is unavailable and the values below come from
#! published specifications instead. Vera Rubin NVL72 quotes 1580 TB/s of aggregate HBM
#! bandwidth across the rack; each package presents two CUDA devices, so the per-device
#! figure is 1580e3 / 144. Entries are added only once confirmed - an unknown key yields
#! no SOL column rather than a fabricated denominator.
KNOWN_PEAK_BANDWIDTH = {
    ("10.7", 286524): 1580e3 / 144,
}


def query_device(device: int) -> dict[str, str]:
    """Return the nvidia-smi properties of `device`, or an empty mapping if unavailable."""
    fields = ("name", "memory.total", "clocks.max.memory", "compute_cap")
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
                f"--id={device}",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    values = [value.strip() for value in output.strip().splitlines()[0].split(",")]
    if len(values) != len(fields):
        return {}
    return dict(zip(fields, values))


def resolve_peak_bandwidth(args: argparse.Namespace) -> dict[str, Any]:
    """Determine the per-device peak bandwidth in GB/s used to compute speed-of-light.

    Resolution order is explicit flag, environment variable, then the table above. The
    result records its own provenance so a summary can always be traced back to the
    number it was normalized against.
    """
    properties = query_device(args.device)
    resolved: dict[str, Any] = {"device": properties, "gb_per_second": None}
    if args.peak_bandwidth is not None:
        resolved |= {"gb_per_second": args.peak_bandwidth, "source": "--peak-bandwidth"}
        return resolved
    override = os.environ.get("CCCL_PEAK_BANDWIDTH_GB_S")
    if override:
        try:
            resolved |= {
                "gb_per_second": float(override),
                "source": "CCCL_PEAK_BANDWIDTH_GB_S",
            }
            return resolved
        except ValueError:
            print(f"ignoring malformed CCCL_PEAK_BANDWIDTH_GB_S={override!r}")
    try:
        key = (properties["compute_cap"], int(properties["memory.total"]))
    except (KeyError, ValueError):
        resolved["source"] = "unavailable (nvidia-smi did not report the device)"
        return resolved
    if key not in KNOWN_PEAK_BANDWIDTH:
        resolved["source"] = (
            f"unavailable (no table entry for compute capability {key[0]} with {key[1]} MiB;"
            " pass --peak-bandwidth)"
        )
        return resolved
    resolved |= {
        "gb_per_second": KNOWN_PEAK_BANDWIDTH[key],
        "source": f"table entry for compute capability {key[0]} with {key[1]} MiB",
    }
    return resolved


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


def summarize(
    results: dict[str, Any], log_dir: pathlib.Path, peak: dict[str, Any]
) -> dict[str, Any]:
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
        baseline = row[BASELINE]["median_seconds"]
        baseline_throughput = row[BASELINE].get("gb_per_second")
        for scenario in SCENARIOS:
            measurement = row[scenario]
            measurement["latency_speedup"] = (
                baseline / measurement["median_seconds"]
                if baseline and measurement["median_seconds"]
                else None
            )
            measurement["throughput_ratio"] = (
                measurement["gb_per_second"] / baseline_throughput
                if baseline_throughput and measurement["median_seconds"]
                else None
            )
            measurement["speed_of_light"] = (
                measurement["gb_per_second"] / peak["gb_per_second"]
                if peak["gb_per_second"] and measurement["median_seconds"]
                else None
            )
        summary_rows.append(row)
    return {
        "log_dir": str(log_dir),
        "scenarios": results,
        "peak_bandwidth": peak,
        "rows": summary_rows,
    }


SPEEDUP_BAR_WIDTH = 12
SOL_BAR_WIDTH = 20


def format_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size:.0f} B"
        size /= 1024
    raise AssertionError("unreachable")


def bar(fraction: float | None, width: int) -> str:
    """Render `fraction` of a bracketed track `width` characters wide.

    A `+` terminates bars that overshoot a whole character by at least half of one, which
    recovers most of the resolution a plain `#` scale would otherwise round away. Values
    are clamped, so a scenario faster than the baseline still reads as a full track.
    """
    if fraction is None:
        return "[" + "?".center(width) + "]"
    filled = max(0.0, min(1.0, fraction)) * width
    whole = int(filled)
    half = "+" if whole < width and filled - whole >= 0.5 else ""
    return "[" + ("#" * whole + half).ljust(width) + "]"


def write_markdown(summary: dict[str, Any], path: pathlib.Path) -> None:
    peak = summary["peak_bandwidth"]
    columns = [
        ("Scenario", "left", max(len(s) for s in SCENARIOS)),
        ("Median ms", "right", 9),
        ("GB/s", "right", 9),
        ("Speedup", "right", 7),
        ("SOL", "right", 6),
    ]
    header = "  ".join(
        title.ljust(width) if align == "left" else title.rjust(width)
        for title, align, width in columns
    )
    header = f"    {header}"
    if peak["gb_per_second"]:
        header += (
            f"  {'vs baseline'.center(SPEEDUP_BAR_WIDTH + 2)}"
            f"  {'vs peak'.center(SOL_BAR_WIDTH + 2)}"
        )
    else:
        header += f"  {'vs baseline'.center(SPEEDUP_BAR_WIDTH + 2)}"

    lines = ["# MGMN benchmark summary", ""]
    device = peak.get("device") or {}
    if device:
        lines.append(
            f"Device {device.get('name', 'unknown')}"
            f" (compute capability {device.get('compute_cap', '?')},"
            f" {device.get('memory.total', '?')} MiB)"
        )
    if peak["gb_per_second"]:
        lines.append(
            f"Peak bandwidth {peak['gb_per_second']:.1f} GB/s"
            f" via {peak['source']}. SOL is the fraction of that peak achieved."
        )
    else:
        lines.append(f"Peak bandwidth {peak['source']}; SOL omitted.")
    lines.extend(["", "```", header, "  " + "-" * (len(header) + 2)])

    for row in summary["rows"]:
        lines.append("")
        lines.append(f"  n = {row['elements']:,}  ({format_bytes(row['input_bytes'])})")
        # Fastest first, so the ranking at each size is apparent without comparing numbers.
        # Scenarios with no measurement sort last; ties keep their declaration order.
        ordered = sorted(
            SCENARIOS,
            key=lambda scenario: (
                row[scenario]["median_seconds"] is None,
                row[scenario]["median_seconds"] or 0.0,
            ),
        )
        # The speedup bar is scaled to the fastest scenario in the group rather than to the
        # baseline: a scenario that beats the baseline exceeds 1.0, and clamping every such case to
        # a full bar would render them indistinguishable from the baseline itself.
        speedups = [
            row[scenario]["latency_speedup"]
            for scenario in ordered
            if row[scenario]["latency_speedup"] is not None
        ]
        speedup_scale = max(speedups) if speedups else 1.0
        for scenario in ordered:
            value = row[scenario]
            if value["median_seconds"] is None:
                cells = [scenario, "n/a", "n/a", "n/a", value["status"]]
                bars = ""
            else:
                sol = value["speed_of_light"]
                cells = [
                    scenario,
                    f"{value['median_seconds'] * 1e3:.3f}",
                    f"{value['gb_per_second']:.3f}",
                    f"{value['latency_speedup']:.3f}",
                    f"{sol * 100:.1f}%" if sol is not None else "n/a",
                ][: len(columns)]
                relative = value["latency_speedup"] / speedup_scale
                bars = (
                    f"  {bar(relative, SPEEDUP_BAR_WIDTH)}  {bar(sol, SOL_BAR_WIDTH)}"
                    if peak["gb_per_second"]
                    else f"  {bar(relative, SPEEDUP_BAR_WIDTH)}"
                )
            body = "  ".join(
                text.ljust(width) if align == "left" else text.rjust(width)
                for text, (_, align, width) in zip(cells, columns)
            )
            lines.append(f"    {body}{bars}")
    lines.append("```")

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
    parser.add_argument(
        "--peak-bandwidth",
        type=float,
        help="per-device peak memory bandwidth in GB/s, used to compute speed-of-light;"
        " overrides autodetection",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sizes = make_sizes(args)
    peak = resolve_peak_bandwidth(args)
    root = pathlib.Path(__file__).resolve().parent
    log_dir = args.log_dir or root / "benchmark-results" / dt.datetime.now(
        dt.timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "sizes": sizes,
        "device": args.device,
        "scenarios": list(SCENARIOS),
        "peak_bandwidth": peak,
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
                #                "NCCL_DEBUG": "INFO" if args.v > 1 else "WARN",
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
        summarize(results, log_dir, peak)
        if not args.dry_run
        else {
            "log_dir": str(log_dir),
            "scenarios": results,
            "peak_bandwidth": peak,
            "rows": [],
        }
    )
    (log_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(summary, log_dir / "summary.md")
    print(log_dir / "summary.md")
    return 0 if all(value["status"] == "ok" for value in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
