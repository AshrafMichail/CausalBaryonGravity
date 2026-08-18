#!/usr/bin/env python3
"""Recompute the flat Causal Baryon validation ledger."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .downloader import download_inputs, verify_present_inputs
from .orchestrator import PredictionOrchestrator
from .registry import CASES, LEDGER_CASES, REFERENCES, VALIDATION_CASES
from .result import Result


HERE = Path(__file__).resolve().parent
CSV_FIELDS = (
    "case", "evidence", "method", "metric", "value", "expected_value",
    "absolute_tolerance", "matches_reference", "difference",
)
JSON_INDENT = 2
NAME_COLUMN_WIDTH = 31
EXIT_COMPLETE, EXIT_INCOMPLETE = 0, 2
# Plot options.
CASE_FIGURE_INCHES = (5.8, 3.8)
CASE_BAR_COLORS = ("#4f46e5", "#94a3b8")
FIGURE_DPI = 150
GRID_ALPHA = 0.2
SUMMARY_MINIMUM_WIDTH_INCHES = 9.0
SUMMARY_WIDTH_PER_CASE_INCHES = 0.55
SUMMARY_HEIGHT_INCHES = 4.8
SUMMARY_LINEAR_THRESHOLD = 1e-12
SUMMARY_LABEL_ROTATION_DEGREES = 60


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=HERE / "runtime_data",
        help="raw-data root; may point to an existing downloaded raw tree",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "runtime_output")
    parser.add_argument(
        "--cases", default="all", help="comma-separated case names, or all",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args(argv)


def _plot(result: Result, output: Path) -> None:
    fig, axis = plt.subplots(figsize=CASE_FIGURE_INCHES)
    axis.bar(
        ["fresh value", "expected value"],
        [result.value, result.expected_value],
        color=list(CASE_BAR_COLORS),
    )
    axis.set_ylabel(result.metric.replace("_", " "))
    axis.set_title(result.name.replace("_", " "))
    axis.grid(axis="y", alpha=GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(output / f"{result.name}.png", dpi=FIGURE_DPI)
    plt.close(fig)


def _write_report(
    results: list[Result], skipped: dict[str, str], mismatches: dict[str, str],
    selected: list[str], output: Path, parameters: dict[str, float | list[float]],
) -> None:
    payload = {
        "model": "Causal Baryon",
        "parameters": parameters,
        "coverage": {
            "paper_rows": len(LEDGER_CASES),
            "extensions": len(VALIDATION_CASES) - len(LEDGER_CASES),
            "selected": len(selected),
            "computed": len(results),
            "skipped": len(skipped),
            "matching": len(results) - len(mismatches),
            "complete": len(results) == len(selected) and not skipped and not mismatches,
        },
        "evidence_counts": dict(Counter(result.evidence for result in results)),
        "results": {result.name: result.summary() for result in results},
        "skipped": skipped,
        "mismatches": mismatches,
        "freshness": (
            "Every number was calculated during this process. No prior "
            "summary or generated result was read."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=JSON_INDENT) + "\n"
    )
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            summary = result.summary()
            tolerance = REFERENCES[result.name][1]
            writer.writerow({
                "case": result.name,
                "evidence": result.evidence,
                "method": result.method,
                "metric": result.metric,
                "value": result.value,
                "expected_value": result.expected_value,
                "absolute_tolerance": tolerance,
                "matches_reference": abs(summary["difference"]) <= tolerance,
                "difference": summary["difference"],
            })
    if results:
        differences = [abs(result.value - result.expected_value) for result in results]
        fig, axis = plt.subplots(
            figsize=(
                max(
                    SUMMARY_MINIMUM_WIDTH_INCHES,
                    len(results) * SUMMARY_WIDTH_PER_CASE_INCHES,
                ),
                SUMMARY_HEIGHT_INCHES,
            )
        )
        axis.bar([result.name for result in results], differences)
        axis.set_yscale("symlog", linthresh=SUMMARY_LINEAR_THRESHOLD)
        axis.set_ylabel("absolute difference from expected value")
        axis.tick_params(axis="x", rotation=SUMMARY_LABEL_ROTATION_DEGREES)
        fig.tight_layout()
        fig.savefig(output / "summary.png", dpi=FIGURE_DPI)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    orchestrator = PredictionOrchestrator()
    bound_cases = {name: spec.bind(orchestrator) for name, spec in CASES.items()}
    selected = (
        list(VALIDATION_CASES)
        if args.cases == "all"
        else [name.strip() for name in args.cases.split(",") if name.strip()]
    )
    unknown = sorted(set(selected) - set(CASES))
    if unknown:
        raise SystemExit(f"unknown cases: {', '.join(unknown)}")
    if args.download or args.force_download:
        download_inputs(args.data_dir, selected, args.force_download)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []
    skipped: dict[str, str] = {}
    mismatches: dict[str, str] = {}
    for name in selected:
        print(f"recomputing {name}")
        try:
            verify_present_inputs(args.data_dir, [name])
            result = bound_cases[name].run(args.data_dir)
        except (FileNotFoundError, ValueError) as error:
            skipped[name] = f"input validation failed: {error}"
            continue
        expected, tolerance = REFERENCES[name]
        if result.expected_value != expected:
            raise ValueError(
                f"{name} returned undeclared reference "
                f"{result.expected_value}; expected {expected}"
            )
        difference = abs(result.value - expected)
        if not np.isfinite(result.value) or difference > tolerance:
            mismatches[name] = (
                f"value {result.value:.17g}, expected {expected:.17g}, "
                f"absolute tolerance {tolerance:.3g}"
            )
        results.append(result)
        _plot(result, args.output_dir)
    _write_report(
        results, skipped, mismatches, selected, args.output_dir,
        orchestrator.parameters(),
    )
    for result in results:
        print(f"{result.name:<{NAME_COLUMN_WIDTH}} {result.value:.10g}")
    for name, reason in skipped.items():
        print(f"{name:<{NAME_COLUMN_WIDTH}} SKIPPED {reason}")
    for name, reason in mismatches.items():
        print(f"{name:<{NAME_COLUMN_WIDTH}} MISMATCH {reason}")
    print(f"wrote {args.output_dir}")
    return EXIT_COMPLETE if not skipped and not mismatches else EXIT_INCOMPLETE


if __name__ == "__main__":
    raise SystemExit(main())
