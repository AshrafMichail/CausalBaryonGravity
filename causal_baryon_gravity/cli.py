"""Command-line interface for the Causal Baryon Gravity effective framework."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmarks import write_summary
from .model import CausalBaryonGravity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser(
        "summary", help="regenerate bundled manuscript summaries and plots"
    )
    summary.add_argument("--output-dir", type=Path, default=Path("outputs"))

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate one spherical weak-field point"
    )
    evaluate.add_argument("--g-b-m-s2", type=float, required=True)
    evaluate.add_argument("--radius-kpc-comoving", type=float, required=True)
    evaluate.add_argument("--radius-kpc-physical", type=float, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "summary":
        summary = write_summary(args.output_dir)
        print(json.dumps(summary, indent=2))
        print(f"wrote {args.output_dir}")
        return 0

    model = CausalBaryonGravity()
    acceleration = float(
        model.acceleration(args.g_b_m_s2, args.radius_kpc_comoving)
    )
    payload = {
        "baryonic_acceleration_m_s2": args.g_b_m_s2,
        "radius_kpc_comoving": args.radius_kpc_comoving,
        "cbg_acceleration_m_s2": acceleration,
        "circular_speed_km_s": float(
            model.circular_speed_km_s(
                args.g_b_m_s2,
                args.radius_kpc_comoving,
                args.radius_kpc_physical,
            )
        ),
    }
    print(json.dumps(payload, indent=2))
    return 0
