"""Physics-free statistics and result construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

from .records import Record
from .result import Result

# Numerical guards and the reported credible quantiles.
PSEUDOINVERSE_RCOND = 1e-12
MINIMUM_NORMALIZER = 1e-30
LOWER_QUANTILE, MEDIAN_QUANTILE, UPPER_QUANTILE = 0.16, 0.50, 0.84
MINIMUM_DISTRIBUTION_INTEGRAL = 0.999
POLE_RESIDUAL_TOLERANCE = 1e-12
DISPERSION_ROOT_COUNT = 2


@dataclass(frozen=True)
class ValidationMetadata:
    name: str
    evidence: str
    method: str
    metric: str
    expected_value: float
    details: dict[str, Any] = field(default_factory=dict)


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(predicted - observed))))


def chi_square(residual: np.ndarray, precision: np.ndarray) -> float:
    return float(residual @ precision @ residual)


def covariance_chi_square(
    observed: np.ndarray, predicted: np.ndarray, covariance: np.ndarray
) -> tuple[float, np.ndarray]:
    precision = np.linalg.inv(covariance)
    return chi_square(observed - predicted, precision), precision


def deviance(log_likelihood: float, reference: float) -> float:
    return float(2.0 * (reference - log_likelihood))


def negative_log_likelihood(residual: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return 0.5 * (np.square(residual / sigma) + np.log(2.0 * np.pi * np.square(sigma)))


def aggregate_log_likelihood(values: np.ndarray) -> float:
    return float(logsumexp(values) - np.log(len(values)))


def correlation(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.corrcoef(observed, predicted)[0, 1])


def relative_residual_norm(residual: np.ndarray, *normalizers: np.ndarray) -> float:
    denominator = 1.0
    for values in normalizers:
        denominator *= np.linalg.norm(values)
    return float(np.linalg.norm(residual) / max(denominator, MINIMUM_NORMALIZER))


def aggregate_flags(flags: dict[str, bool]) -> float:
    return float(all(flags.values()))


def _result(
    metadata: ValidationMetadata, value: float, details: dict[str, Any] | None = None
) -> Result:
    return Result(
        metadata.name, metadata.evidence, metadata.method, metadata.metric,
        float(value), metadata.expected_value,
        {**metadata.details, **(details or {})},
    )


def validate_scalar(prediction: Record, metadata: ValidationMetadata) -> Result:
    return _result(metadata, prediction["value"], prediction["details"])


def validate_galaxy(prediction: Record, metadata: ValidationMetadata) -> Result:
    """Compare common-input galaxy predictions using velocity RMSE."""
    observed = prediction["observed_speed_km_s"]
    speeds = prediction["model_speeds_km_s"]
    causal_rmse = rmse(observed, speeds["Causal Baryon"])
    galaxies = prediction["galaxies"]
    return _result(
        metadata, causal_rmse,
        {
            "sources": list(prediction["sources"]),
            "galaxies": int(np.unique(galaxies).size),
            "radial_points": int(len(galaxies)),
            "galaxies_with_external_field": int(
                np.unique(galaxies[prediction["external_acceleration_m_s2"] > 0.0]).size
            ),
            "causal_baryon_rmse_km_s": causal_rmse,
            "fixed_simple_mond_rmse_km_s": rmse(observed, speeds["Simple MOND"]),
            "baryons_only_rmse_km_s": rmse(observed, speeds["Baryons"]),
            "external_field_method": (
                "Per-galaxy maximum-clustering values extracted locally from "
                "Chae et al. (2021), with zero for unmatched galaxies."
            ),
        },
    )


def validate_planck_spectra(prediction: Record, metadata: ValidationMetadata) -> Result:
    """Profile calibration and evaluate the full Planck spectral covariance."""
    observed = prediction["observed"]
    predicted = prediction["predicted"]
    count = len(observed)
    precision = cho_solve(
        cho_factor(prediction["covariance"], lower=True), np.eye(count)
    )
    calibration_sigma = prediction["calibration_sigma"]
    prior_precision = 1.0 / calibration_sigma**2
    scale = float(
        (predicted @ precision @ observed + prior_precision)
        / (predicted @ precision @ predicted + prior_precision)
    )
    residual = observed - scale * predicted
    chi2 = chi_square(residual, precision) + ((scale - 1.0) / calibration_sigma) ** 2
    return _result(
        metadata, chi2 / count,
        {
            "source": prediction["source"],
            "bins": count,
            "chi2": float(chi2),
            "profiled_power_calibration": scale,
            "ontology_note": (
                "At this endpoint the response equations are identical to "
                "the reference pressureless CAMB component."
            ),
        },
    )


def validate_planck_lensing(prediction: Record, metadata: ValidationMetadata) -> Result:
    chi2, precision = covariance_chi_square(
        prediction["observed"], prediction["predicted"], prediction["covariance"]
    )
    predicted = prediction["predicted"]
    amplitude = float(
        predicted @ precision @ prediction["observed"]
        / (predicted @ precision @ predicted)
    )
    return _result(
        metadata, chi2 / len(prediction["observed"]),
        {
            "source": prediction["source"],
            "bands": len(prediction["observed"]),
            "chi2": chi2,
            "A_phi": amplitude,
        },
    )


def validate_sdss(prediction: Record, metadata: ValidationMetadata) -> Result:
    total_chi2 = 0.0
    total_points = 0
    rows = []
    for item in prediction["tracers"]:
        residual = item["observed"] - item["predicted"]
        precision = np.linalg.pinv(
            item["covariance"], hermitian=True, rcond=PSEUDOINVERSE_RCOND
        )
        item_chi2 = chi_square(residual, precision)
        total_chi2 += item_chi2
        total_points += len(residual)
        rows.append(
            {"tracer": item["tracer"], "points": len(residual), "chi2": item_chi2}
        )
    return _result(
        metadata, total_chi2 / total_points,
        {
            "source": prediction["source"],
            "points": total_points,
            "chi2": total_chi2,
            "per_tracer": rows,
        },
    )


def validate_bullet_apertures(prediction: Record, metadata: ValidationMetadata) -> Result:
    log_likelihood = np.zeros(prediction["samples"])
    medians = []
    ratio_medians = []
    for index, values in enumerate(prediction["predicted_plasma_kappa"]):
        sigma = np.hypot(
            prediction["plasma_errors"][index],
            prediction["fractional_prediction_error"] * np.abs(values),
        )
        log_likelihood -= negative_log_likelihood(
            prediction["observed_plasma_kappa"][index] - values, sigma
        )
        medians.append(float(np.median(values)))
        ratio_medians.append(
            float(np.median(prediction["bcg_to_plasma_ratio"][index]))
        )
    return _result(
        metadata, medians[0],
        {
            "source": prediction["source"],
            "aperture_radius_kpc": prediction["aperture_radius_kpc"],
            "subcluster_predicted_plasma_kappa": medians[1],
            "main_bcg_to_plasma_ratio": ratio_medians[0],
            "subcluster_bcg_to_plasma_ratio": ratio_medians[1],
            "marginal_log_likelihood": aggregate_log_likelihood(log_likelihood),
            "collisionless_to_baryon_ratio": (
                prediction["collisionless_to_baryon_ratio"]
            ),
            "samples": prediction["samples"],
            "random_seed": prediction["random_seed"],
            "limitation": (
                "Published aperture central values and errors are embedded; "
                "this is not a convergence-map likelihood."
            ),
        },
    )


def _cluster_rows(prediction: Record) -> list[Record]:
    rows = []
    for held in prediction["held_systems"]:
        candidates = held["candidates"]
        scores = {
            candidate["name"]: float(
                np.mean([
                    rmse(item["observed"], item["predicted"])
                    for item in candidate["nested"]
                ])
            )
            for candidate in candidates
        }
        chosen_name = min(scores, key=scores.get)
        chosen = next(item for item in candidates if item["name"] == chosen_name)
        for item in chosen["held"]:
            rows.append({
                "held_system": held["held_system"],
                "lensing_reconstruction": item["reconstruction"],
                "selected_model": chosen_name,
                "pearson_correlation": correlation(item["observed"], item["predicted"]),
                "standardized_rmse": rmse(item["observed"], item["predicted"]),
            })
    return rows


def validate_cluster(prediction: Record, metadata: ValidationMetadata) -> Result:
    """Select nested proxy scales and summarize held-cluster correlations."""
    rows = _cluster_rows(prediction)
    if metadata.name == "hff_lens_robustness":
        selected = [row for row in rows if row["held_system"] != "abell520"]
        value = min(row["pearson_correlation"] for row in selected)
    elif metadata.name == "abell520_morphology":
        selected = [row for row in rows if row["held_system"] == "abell520"]
        value = min(row["pearson_correlation"] for row in selected)
    else:
        selected = rows
        value = float(pd.Series([row["pearson_correlation"] for row in rows])
                      .groupby(lambda _: 0).mean().iloc[0])
    return _result(
        metadata, value,
        {
            "source": prediction["source"],
            "rows": selected,
            "shared_calculation": "cluster_map_nested_cross_validation",
            "limitation": (
                "Light and X-ray intensity are morphology proxies, not "
                "calibrated baryonic stress-energy maps."
            ),
        },
    )


def _profile_log_likelihood(
    sigma: np.ndarray, velocity: np.ndarray, error: np.ndarray
) -> np.ndarray:
    proposed = np.atleast_1d(sigma)[:, None]
    variance = np.square(error)[None, :] + np.square(proposed)
    weight = 1.0 / variance
    mean = np.sum(weight * velocity[None, :], axis=1) / np.sum(weight, axis=1)
    return -0.5 * np.sum(
        np.log(2.0 * np.pi * variance)
        + np.square(velocity[None, :] - mean[:, None]) / variance,
        axis=1,
    )


def validate_df4(prediction: Record, metadata: ValidationMetadata) -> Result:
    """Marginalize nuisance draws and compare DF4 models by deviance."""
    rows: dict[str, Record] = {}
    for name, sigma in prediction["sigma_samples_km_s"].items():
        likelihood = _profile_log_likelihood(
            sigma, prediction["velocities_km_s"], prediction["errors_km_s"]
        )
        rows[name] = {
            "sigma_p16_km_s": float(np.quantile(sigma, LOWER_QUANTILE)),
            "sigma_median_km_s": float(np.quantile(sigma, MEDIAN_QUANTILE)),
            "sigma_p84_km_s": float(np.quantile(sigma, UPPER_QUANTILE)),
            "marginal_log_likelihood": aggregate_log_likelihood(likelihood),
            "best_log_likelihood": float(np.max(likelihood)),
        }
    reference = max(row["marginal_log_likelihood"] for row in rows.values())
    for row in rows.values():
        row["delta_deviance_from_best"] = deviance(
            row["marginal_log_likelihood"], reference
        )
    value = rows["Causal Baryon Model"]["delta_deviance_from_best"]
    canonical = rows["Canonical NFW"]["delta_deviance_from_best"]
    return _result(
        metadata, value,
        {
            "source": prediction["source"],
            "globular_clusters": prediction["tracer_count"],
            "monte_carlo_samples": prediction["samples"],
            "random_seed": prediction["random_seed"],
            "models": rows,
            "canonical_nfw_delta_deviance": canonical,
            "flexible_nfw_delta_deviance": rows["Flexible NFW"][
                "delta_deviance_from_best"
            ],
            "outperforms_canonical_nfw": value < canonical,
            "scope": (
                "The flexible-NFW comparison is prior dependent and does not "
                "exclude stripped or unusually low-mass halos."
            ),
        },
    )


def validate_bianchi(prediction: Record, metadata: ValidationMetadata) -> Result:
    maximum = 0.0
    for wave, response in zip(prediction["wavevectors"], prediction["responses"]):
        divergence = np.einsum("m,mn->n", wave, response)
        maximum = max(maximum, relative_residual_norm(divergence, wave, response))
    return _result(metadata, maximum)


def validate_gate(prediction: Record, metadata: ValidationMetadata) -> Result:
    inactive, active_one, active_two = prediction["responses"]
    flags = (
        (inactive <= prediction["maximum_inactive_response"])
        & (active_one >= prediction["minimum_active_response"])
        & (active_two >= prediction["minimum_active_response"])
    )
    return _result(
        metadata, float(np.count_nonzero(flags)), {"total_grid_points": int(inactive.size)}
    )


def validate_stability(prediction: Record, metadata: ValidationMetadata) -> Result:
    roots = prediction["dispersion_roots"]
    probe_waves = prediction["probe_wave_numbers"]
    pole_residual = max(
        float(np.max(np.abs(values**2 - wave_number**2)))
        for values, wave_number in zip(roots, probe_waves)
    )
    distribution_integral = float(
        np.trapz(prediction["distribution"], prediction["momentum"])
    )
    flags = {
        "positive_distribution": bool(
            np.all(np.isfinite(prediction["distribution"]))
            and np.all(prediction["distribution"] >= 0.0)
            and distribution_integral > MINIMUM_DISTRIBUTION_INTEGRAL
        ),
        "positive_amplitude": prediction["response_amplitude"] > 0.0,
        "bounded_gate": bool(
            np.all((prediction["gate"] >= 0.0) & (prediction["gate"] <= 1.0))
        ),
        "positive_local_response": bool(np.all(prediction["susceptibility"] >= 0.0)),
        "no_upper_half_plane_tensor_poles": bool(
            np.max(prediction["tensor_poles"].imag) <= np.finfo(float).eps
        ),
        "no_superluminal_characteristics": bool(
            np.max(prediction["characteristic_speeds"]) <= 1.0
        ),
        "no_extra_radiative_poles": bool(
            all(len(values) == DISPERSION_ROOT_COUNT for values in roots)
            and pole_residual
            <= POLE_RESIDUAL_TOLERANCE * np.max(np.square(probe_waves))
        ),
    }
    return _result(
        metadata, aggregate_flags(flags),
        {
            **flags,
            "distribution_integral": distribution_integral,
            "tensor_pole_max_imaginary_part": float(
                np.max(prediction["tensor_poles"].imag)
            ),
            "maximum_characteristic_speed_c": float(
                np.max(prediction["characteristic_speeds"])
            ),
            "maximum_dispersion_polynomial_residual": pole_residual,
        },
    )


def validate_particle_mesh(prediction: Record, metadata: ValidationMetadata) -> Result:
    """Compare density and power outputs from the two particle representations."""
    reference = prediction["densities"]["Reference"]
    response = prediction["densities"]["Causal Baryon"]
    value = float(np.max(np.abs(reference - response)))
    reference_power = np.abs(np.fft.rfft(reference)) ** 2
    response_power = np.abs(np.fft.rfft(response)) ** 2
    power_difference = float(
        np.max(np.abs(reference_power - response_power))
        / max(float(np.max(reference_power)), MINIMUM_NORMALIZER)
    )
    return _result(
        metadata, value,
        {
            "particles_per_component": prediction["particles_per_component"],
            "mesh_cells": prediction["mesh_cells"],
            "steps": prediction["steps"],
            "maximum_relative_power_difference": power_difference,
            "reference_path": "reference two-component evolution",
            "response_path": "Causal Baryon two-component evolution",
        },
    )
