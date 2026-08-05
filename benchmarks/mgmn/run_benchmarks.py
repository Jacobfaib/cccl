#!/usr/bin/env python3
"""Run and summarize the independently built MGMN reduction benchmarks.

Each scenario is a separate NVBench executable, so NVBench cannot rank them against each
other. This script runs all four, reads their JSON output, and emits a per-size comparison
against the `cub_full_device` baseline.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import subprocess
from typing import Any

SCENARIOS = (
    "cub_full_device",
    "cudax_host_nccl",
    "cub_green_atomic",
    #    "cudax_device_nccl",
)

BASELINE = SCENARIOS[0]

#! NVBench summary tags read out of each state. NVBench derives the bandwidth figures from
#! the element counts and byte volumes the benchmarks declare, and normalizes utilization
#! against the device's own peak bandwidth, so no peak-bandwidth table is needed here.
GPU_TIME_TAG = "nv/cold/time/gpu/mean"
GPU_NOISE_TAG = "nv/cold/time/gpu/stdev/relative"
BANDWIDTH_TAG = "nv/cold/bw/global/bytes_per_second"
UTILIZATION_TAG = "nv/cold/bw/global/utilization"
SAMPLES_TAG = "nv/cold/sample_size"


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


def make_axis_override(args: argparse.Namespace) -> str | None:
    """Build the NVBench `-a` argument that overrides the `Elements` axis.

    Returns None when the defaults compiled into the benchmarks are wanted. The axis is
    declared power-of-two, so explicit sizes are passed as exponents and must therefore be
    exact powers of two.
    """
    if args.sizes:
        exponents = []
        for size in args.sizes:
            if size & (size - 1):
                raise ValueError(
                    f"size {size} is not a power of two; the Elements axis is pow2"
                )
            exponents.append(size.bit_length() - 1)
        return "Elements[pow2]=[" + ",".join(map(str, exponents)) + "]"
    if (
        args.min_elements_pow2 is None
        and args.max_elements_pow2 is None
        and args.stride is None
    ):
        return None
    low = args.min_elements_pow2 if args.min_elements_pow2 is not None else 20
    high = args.max_elements_pow2 if args.max_elements_pow2 is not None else 28
    stride = args.stride if args.stride is not None else 2
    if low <= 0 or high < low or stride < 1:
        raise ValueError("invalid element exponent range")
    return f"Elements[pow2]=[{low}:{high}:{stride}]"


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


def summary_value(state: dict[str, Any], tag: str) -> float | None:
    """Read the scalar `value` of the summary tagged `tag` out of one NVBench state.

    NVBench writes every summary datum as a `{name, type, value}` triple with the value
    rendered as a string, so the number is recovered here rather than taken as-is.
    """
    for summary in state.get("summaries", []):
        if summary.get("tag") != tag:
            continue
        for datum in summary.get("data", []):
            if datum.get("name") == "value":
                try:
                    return float(datum["value"])
                except (KeyError, TypeError, ValueError):
                    return None
    return None


def axis_value(state: dict[str, Any], name: str) -> int | None:
    for value in state.get("axis_values", []):
        if value.get("name") == name:
            try:
                return int(float(value["value"]))
            except (KeyError, TypeError, ValueError):
                return None
    return None


def collect(document: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Reduce one NVBench JSON document to a measurement per element count."""
    measurements: dict[int, dict[str, Any]] = {}
    for benchmark in document.get("benchmarks", []):
        for state in benchmark.get("states", []):
            elements = axis_value(state, "Elements")
            if elements is None:
                continue
            if state.get("is_skipped"):
                measurements[elements] = {
                    "status": f"skipped ({state.get('skip_reason', 'no reason given')})",
                    "seconds": None,
                }
                continue
            seconds = summary_value(state, GPU_TIME_TAG)
            if seconds is None:
                continue
            measurements[elements] = {
                "status": "ok",
                "seconds": seconds,
                "noise": summary_value(state, GPU_NOISE_TAG),
                "samples": summary_value(state, SAMPLES_TAG),
                "gb_per_second": (
                    value / 1e9
                    if (value := summary_value(state, BANDWIDTH_TAG)) is not None
                    else None
                ),
                "utilization": summary_value(state, UTILIZATION_TAG),
            }
    return measurements


def summarize(results: dict[str, Any], log_dir: pathlib.Path) -> dict[str, Any]:
    per_scenario = {
        scenario: collect(result["json"])
        for scenario, result in results.items()
        if result.get("status") == "ok" and "json" in result
    }
    elements_seen = sorted(
        {elements for values in per_scenario.values() for elements in values}
    )

    rows = []
    for elements in elements_seen:
        row: dict[str, Any] = {"elements": elements, "input_bytes": elements * 4}
        for scenario in SCENARIOS:
            row[scenario] = per_scenario.get(scenario, {}).get(
                elements,
                {"status": results[scenario]["status"], "seconds": None},
            )
        baseline = row[BASELINE]["seconds"]
        for scenario in SCENARIOS:
            measurement = row[scenario]
            measurement["latency_speedup"] = (
                baseline / measurement["seconds"]
                if baseline and measurement["seconds"]
                else None
            )
        rows.append(row)
    return {"log_dir": str(log_dir), "scenarios": results, "rows": rows}


SPEEDUP_BAR_WIDTH = 12
UTILIZATION_BAR_WIDTH = 20


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
    columns = [
        ("Scenario", "left", max(len(s) for s in SCENARIOS)),
        ("GPU ms", "right", 9),
        ("GB/s", "right", 9),
        ("Speedup", "right", 7),
        ("BWUtil", "right", 6),
    ]
    header = "  ".join(
        title.ljust(width) if align == "left" else title.rjust(width)
        for title, align, width in columns
    )
    header = (
        f"    {header}"
        f"  {'vs baseline'.center(SPEEDUP_BAR_WIDTH + 2)}"
        f"  {'vs peak'.center(UTILIZATION_BAR_WIDTH + 2)}"
    )

    lines = [
        "# MGMN benchmark summary",
        "",
        "GPU times are NVBench cold-measurement means. BWUtil is NVBench's global memory",
        "bandwidth utilization, normalized against the device's own peak bandwidth.",
        "",
        "```",
        header,
        "  " + "-" * (len(header) + 2),
    ]

    rows = []
    for row in summary["rows"]:
        lines.append("")
        lines.append(f"  n = {row['elements']:,}  ({format_bytes(row['input_bytes'])})")
        # Fastest first, so the ranking at each size is apparent without comparing numbers.
        # Scenarios with no measurement sort last; ties keep their declaration order.
        ordered = sorted(
            SCENARIOS,
            key=lambda scenario: (
                row[scenario]["seconds"] is None,
                row[scenario]["seconds"] or 0.0,
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
            if value["seconds"] is None:
                cells = [scenario, "n/a", "n/a", "n/a", "n/a"]
                bars = f"  {value['status']}"
            else:
                utilization = value.get("utilization")
                bandwidth = value.get("gb_per_second")
                # NVBench reports utilization as a fraction of peak, not a percentage.
                cells = [
                    scenario,
                    f"{value['seconds'] * 1e3:.3f}",
                    f"{bandwidth:.3f}" if bandwidth is not None else "n/a",
                    f"{value['latency_speedup']:.3f}",
                    f"{utilization * 100.0:.1f}%" if utilization is not None else "n/a",
                ]
                relative = value["latency_speedup"] / speedup_scale
                bars = (
                    f"  {bar(relative, SPEEDUP_BAR_WIDTH)}"
                    f"  {bar(utilization, UTILIZATION_BAR_WIDTH)}"
                )

    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
        for i, h in enumerate(headers)
    ]

    def fmt(cells):
        return (
            "| "
            + " | ".join(
                c.rjust(w) if a == "right" else c.ljust(w)
                for c, w, a in zip(cells, widths, aligns)
            )
            + " |"
        )

    sep = (
        "|"
        + "|".join(
            ("-" * (w + 1) + ":") if a == "right" else (":" + "-" * (w + 1))
            for w, a in zip(widths, aligns)
        )
        + "|"
    )

    lines = ["# MGMN benchmark summary", "", fmt(headers), sep]
    lines.extend(fmt(r) for r in rows)
    # lines = [
    #     "# MGMN benchmark summary",
    #     "",
    #     "| Elements | Scenario | Median ms | GB/s | Speedup | Status |",
    #     "|---------:|----------|----------:|-----:|--------:|--------|",
    # ]
    # for row in summary["rows"]:
    #     for scenario in SCENARIOS:
    #         value = row[scenario]
    #         if value["median_seconds"] is None:
    #             lines.append(
    #                 f"| {row['elements']} | {scenario} | N/A | N/A | N/A | {value['status']} |"
    #             )
    #         else:
    #             lines.append(
    #                 f"| {row['elements']} | {scenario} | {value['median_seconds'] * 1e3:.3f} | {value['gb_per_second']:.3f} | {value['latency_speedup']:.3f} | ok |"
    #             )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n")
    print(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", action="count", default=0)
    parser.add_argument(
        "--binary-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent.parent.parent
        / "build"
        / "preset-latest"
        / "benchmarks"
        / "mgmn",
    )
    parser.add_argument("--log-dir", type=pathlib.Path)
    parser.add_argument("--device", type=int, default=0)
    # The benchmarks compile in a 2^20..2^28 sweep. These flags override that axis; leaving
    # them unset runs the compiled-in default.
    parser.add_argument("--min-elements-pow2", type=int)
    parser.add_argument("--max-elements-pow2", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        help="explicit element counts; each must be a power of two",
    )
    parser.add_argument("--min-time", type=float, default=0.5)
    parser.add_argument("--max-noise", type=float)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    axis = make_axis_override(args)
    root = pathlib.Path(__file__).resolve().parent
    log_dir = args.log_dir or root / "benchmark-results" / dt.datetime.now(
        dt.timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "axis": axis,
        "device": args.device,
        "scenarios": list(SCENARIOS),
        "commands": [],
    }
    results: dict[str, Any] = {}
    for scenario in SCENARIOS:
        json_path = log_dir / f"{scenario}.json"
        executable = args.binary_dir / scenario
        if not args.dry_run and not executable.is_file():
            raise FileNotFoundError(f"missing benchmark executable: {executable}")
        # `CUDA_VISIBLE_DEVICES` already restricts the process to one GPU, which NVBench then
        # sees as device 0. Without `-d` NVBench would sweep every device it can see.
        command = [
            str(executable),
            "--json",
            str(json_path),
            "-d",
            "0",
            "--min-time",
            str(args.min_time),
        ]
        if axis is not None:
            command += ["-a", axis]
        if args.max_noise is not None:
            command += ["--max-noise", str(args.max_noise)]
        if args.timeout is not None:
            command += ["--timeout", str(args.timeout)]
        manifest["commands"].append(command)
        env = os.environ | {"CUDA_VISIBLE_DEVICES": str(args.device)}
        if "nccl" in scenario:
            env |= {
                "NCCL_MULTI_RANK_GPU_ENABLE": "1",
                "NCCL_NVLS_ENABLE": "0",
                "NCCL_MAX_CTAS": "1",
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
