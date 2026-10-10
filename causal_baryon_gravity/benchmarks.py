"""Reproduce manuscript headline summaries from versioned reduced inputs."""

from __future__ import annotations

import csv
import json
from decimal import Decimal, localcontext
from importlib.resources import files
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .model import CausalBaryonGravity


def _inputs() -> dict[str, object]:
    path = files("causal_baryon_gravity").joinpath("data/headline_inputs.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _rmse(squared_error_sum: str, count: int) -> float:
    with localcontext() as context:
        context.prec = 40
        return float((Decimal(squared_error_sum) / Decimal(count)).sqrt())


def headline_summary() -> dict[str, object]:
    """Calculate all headline values available from the bundled reduced inputs."""
    inputs = _inputs()
    sparc = inputs["sparc"]
    bullet = inputs["bullet"]
    clusters = inputs["clusters"]
    cosmology = inputs["cosmology_transfer_adapter"]
    df = inputs["df2_df4"]
    model = CausalBaryonGravity()

    omega_nu = cosmology["neutrino_mass_ev"] / cosmology["ev_per_omega_nu"]
    response_fraction = cosmology["omega_resp"] / (
        cosmology["omega_b"] + cosmology["omega_resp"] + omega_nu
    )
    return {
        "framework": "Causal Baryon Gravity effective framework",
        "parameters": {
            "a0_m_s2": model.a0_m_s2,
            "A_mem": model.memory_amplitude,
            "sigma_h_kpc_comoving": model.sigma_h_kpc_comoving,
        },
        "sparc": {
            "rmse_km_s": _rmse(
                sparc["cbg_squared_error_sum_km2_s2"], sparc["radii"]
            ),
            "canonical_only_rmse_km_s": _rmse(
                sparc["canonical_squared_error_sum_km2_s2"], sparc["radii"]
            ),
            "radii": sparc["radii"],
            "galaxies": sparc["galaxies"],
            "input_level": "reduced sufficient statistics",
        },
        "bullet": {
            "response_to_baryon_normalization": (
                model.response_to_baryon_normalization
            ),
            "centroid_kpc": bullet["centroid_numerator_kpc"]
            / bullet["centroid_denominator"],
            "aperture_deviance": float(
                np.sum(bullet["aperture_deviance_contributions"])
            ),
            "input_level": "reduced domain-adapter outputs",
        },
        "cluster_maps": {
            "mean": float(np.mean(clusters["held_map_correlations"])),
            "hff_minimum": float(np.min(clusters["hff_correlations"])),
            "a520_minimum": float(np.min(clusters["a520_correlations"])),
            "input_level": "reduced morphology-adapter outputs",
        },
        "cosmology_transfer_estimates": {
            "omega_b": cosmology["omega_b"],
            "omega_resp": cosmology["omega_resp"],
            "omega_nu": omega_nu,
            "response_fraction": response_fraction,
            "interpretation": (
                "transfer-adapter estimates, not predictions of a derived "
                "baryon-only cosmology"
            ),
        },
        "limitations": {
            "df2_fixed_law_dispersion_km_s": float(
                np.sqrt(df["df2_fixed_law_variance_km2_s2"])
            ),
            "df4_fixed_law_dispersion_km_s": float(
                np.sqrt(df["df4_fixed_law_variance_km2_s2"])
            ),
            "interpretation": (
                "less discrepant than canonical abundance-matched NFW, "
                "but worse than baryons alone"
            ),
        },
        "claim_boundary": (
            "The implemented claim is the closed-form spherical weak-field law. "
            "Causal tensor memory is a proposed completion and interpretation."
        ),
    }


def write_summary(output_dir: Path) -> dict[str, object]:
    """Write JSON, CSV, and plots for the bundled summary and model profile."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = headline_summary()
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    rows = [
        ("sparc_rmse_km_s", summary["sparc"]["rmse_km_s"]),
        (
            "sparc_canonical_only_rmse_km_s",
            summary["sparc"]["canonical_only_rmse_km_s"],
        ),
        (
            "bullet_response_to_baryon_normalization",
            summary["bullet"]["response_to_baryon_normalization"],
        ),
        ("bullet_centroid_kpc", summary["bullet"]["centroid_kpc"]),
        ("bullet_aperture_deviance", summary["bullet"]["aperture_deviance"]),
        ("cluster_map_mean", summary["cluster_maps"]["mean"]),
        ("hff_minimum", summary["cluster_maps"]["hff_minimum"]),
        ("a520_minimum", summary["cluster_maps"]["a520_minimum"]),
        (
            "cosmology_transfer_response_fraction",
            summary["cosmology_transfer_estimates"]["response_fraction"],
        ),
        (
            "df2_fixed_law_dispersion_km_s",
            summary["limitations"]["df2_fixed_law_dispersion_km_s"],
        ),
        (
            "df4_fixed_law_dispersion_km_s",
            summary["limitations"]["df4_fixed_law_dispersion_km_s"],
        ),
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]
    axis.bar(labels, values, color="#315a8a")
    axis.set_yscale("symlog", linthresh=1.0)
    axis.set_ylabel("reported value (mixed units; see labels)")
    axis.tick_params(axis="x", rotation=65)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "headline_summary.png", dpi=160)
    plt.close(figure)

    model = CausalBaryonGravity()
    radius = np.geomspace(0.1, 1000.0, 300)
    baryonic = np.full_like(radius, 1.0e-11)
    acceleration = model.acceleration(baryonic, radius)
    speed = model.circular_speed_km_s(baryonic, radius, radius)
    np.savetxt(
        output_dir / "acceleration_profile.csv",
        np.column_stack((radius, baryonic, acceleration, speed)),
        delimiter=",",
        header=(
            "radius_kpc_comoving,baryonic_acceleration_m_s2,"
            "cbg_acceleration_m_s2,circular_speed_km_s_z0_example"
        ),
        comments="",
    )
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.loglog(radius, acceleration, label="g_CBG")
    axis.loglog(radius, baryonic, linestyle="--", label="g_b")
    axis.set_xlabel("comoving radius [kpc]")
    axis.set_ylabel("acceleration [m/s^2]")
    axis.grid(which="both", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "acceleration_profile.png", dpi=160)
    plt.close(figure)
    return summary
