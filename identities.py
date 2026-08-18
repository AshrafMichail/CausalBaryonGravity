"""Adapters for published summaries and deterministic synthetic probes.

Every embedded published measurement and every synthetic probe setting is a
named module constant here; the adapters below only package them.
"""

from pathlib import Path

import numpy as np

from .records import BULLET_APERTURE_COLUMNS, Record

# Bullet Cluster, Clowe et al. (2006).
BULLET_STELLAR_FRACTION = 0.15
BULLET_STELLAR_POSITION_KPC = 100.0
BULLET_PLASMA_POSITION_KPC = 0.0
BULLET_APERTURE_RADIUS_KPC = 100.0
BULLET_SYSTEM_NAMES = ("main", "subcluster")
BULLET_POSITION_NAMES = ("BCG", "plasma")
# One row per (system, aperture position) in BULLET_APERTURE_COLUMNS order.
BULLET_APERTURE_ROWS = (
    ((5.5, 0.6, 0.54, 0.08, 0.36, 0.06), (6.6, 0.7, 0.23, 0.02, 0.05, 0.06)),
    ((2.7, 0.3, 0.58, 0.09, 0.20, 0.05), (5.8, 0.6, 0.12, 0.01, 0.02, 0.06)),
)
BULLET_SAMPLES = 100_000
BULLET_RANDOM_SEED = 20250308
BULLET_FRACTIONAL_PREDICTION_ERROR = 0.10
BULLET_APERTURE_SOURCE = "Clowe et al. (2006), Table 2"

# Big-bang nucleosynthesis background, radiation density and epoch.
BBN_RADIATION_DENSITY_H2 = 9.2e-5
BBN_REDSHIFT = 1.0e9

# Sgr A*, GRAVITY Collaboration (2022) scale and EHT (2022) ring diameter.
SGR_A_MASS_MSUN = 4.297e6
SGR_A_DISTANCE_PC = 8277.0
SGR_A_RING_DIAMETER_MICROARCSEC = 51.8

# Deterministic synthetic probes: Monte Carlo sizes, seeds, and grids given as
# (start, stop, count) arguments for the generator named at each use.
BIANCHI_SAMPLES = 1000
BIANCHI_RANDOM_SEED = 23
GATE_THRESHOLD_GRID = (1e-7, 1e-4, 241)
GATE_POWER_GRID = (1.0, 12.0, 221)
GATE_GALAXY_INVARIANT = 1.218299465866705e-6
GATE_CMB_INVARIANT = 1e-5
GATE_CLUSTER_SPEED_KM_S = 1000.0
GATE_MAXIMUM_INACTIVE_RESPONSE = 0.01
GATE_MINIMUM_ACTIVE_RESPONSE = 0.99
STABILITY_MOMENTUM_GRID = (-8.0, 8.0, 4001)
STABILITY_INVARIANT_GRID = (1e-12, 1e-2, 2000)
STABILITY_ACCELERATION_GRID_M_S2 = (1e-14, 1e2, 2000)
STABILITY_WAVE_NUMBER_GRID = (1e-6, 1e6, 2000)
STABILITY_PROBE_WAVE_NUMBER_GRID = (1e-6, 1e6, 64)
MESH_PARTICLES = 4096
MESH_CELLS = 1024
MESH_BOX_LENGTH = 2.0 * np.pi
MESH_PERTURBATION_AMPLITUDE = 0.02
MESH_TIME_STEP = 0.01
MESH_STEPS = 150


def bullet_centroid(_: Path) -> Record:
    return {
        "stellar_fraction": BULLET_STELLAR_FRACTION,
        "stellar_position_kpc": BULLET_STELLAR_POSITION_KPC,
        "plasma_position_kpc": BULLET_PLASMA_POSITION_KPC,
    }


def bullet_aperture_peaks(_: Path) -> Record:
    return {
        "system_names": BULLET_SYSTEM_NAMES,
        "position_names": BULLET_POSITION_NAMES,
        "measurements": np.asarray(BULLET_APERTURE_ROWS, dtype=float),
        "columns": BULLET_APERTURE_COLUMNS,
        "aperture_radius_kpc": BULLET_APERTURE_RADIUS_KPC,
        "samples": BULLET_SAMPLES,
        "random_seed": BULLET_RANDOM_SEED,
        "fractional_prediction_error": BULLET_FRACTIONAL_PREDICTION_ERROR,
        "source": BULLET_APERTURE_SOURCE,
    }


def _no_input_identity(_: Path) -> Record:
    """Identity rows follow from the equations and read no released input."""
    return {}


lensing_consistency = _no_input_identity
cmb_isocurvature = _no_input_identity
ppn = _no_input_identity
gw_propagation = _no_input_identity
binary_orbital_decay = _no_input_identity


def bbn_background(_: Path) -> Record:
    return {
        "radiation_density_h2": BBN_RADIATION_DENSITY_H2,
        "redshift": BBN_REDSHIFT,
    }


def sgr_a(_: Path) -> Record:
    return {
        "mass_msun": SGR_A_MASS_MSUN,
        "distance_pc": SGR_A_DISTANCE_PC,
        "observed_ring_diameter_microarcsec": SGR_A_RING_DIAMETER_MICROARCSEC,
    }


def bianchi_transversality(_: Path) -> Record:
    return {"samples": BIANCHI_SAMPLES, "random_seed": BIANCHI_RANDOM_SEED}


def gate_robustness(_: Path) -> Record:
    return {
        "thresholds": np.geomspace(*GATE_THRESHOLD_GRID),
        "powers": np.linspace(*GATE_POWER_GRID),
        "galaxy_invariant": GATE_GALAXY_INVARIANT,
        "cmb_invariant": GATE_CMB_INVARIANT,
        "cluster_speed_km_s": GATE_CLUSTER_SPEED_KM_S,
        "maximum_inactive_response": GATE_MAXIMUM_INACTIVE_RESPONSE,
        "minimum_active_response": GATE_MINIMUM_ACTIVE_RESPONSE,
    }


def linear_stability(_: Path) -> Record:
    return {
        "momentum": np.linspace(*STABILITY_MOMENTUM_GRID),
        "invariants": np.geomspace(*STABILITY_INVARIANT_GRID),
        "local_accelerations_m_s2": np.geomspace(*STABILITY_ACCELERATION_GRID_M_S2),
        "wave_numbers": np.geomspace(*STABILITY_WAVE_NUMBER_GRID),
        "probe_wave_numbers": np.geomspace(*STABILITY_PROBE_WAVE_NUMBER_GRID),
    }


def nonlinear_pm_equivalence(_: Path) -> Record:
    """Seed both particle representations with one sine displacement field."""
    q = (np.arange(MESH_PARTICLES) + 0.5) / MESH_PARTICLES * MESH_BOX_LENGTH
    displacement = MESH_PERTURBATION_AMPLITUDE * np.sin(q)
    return {
        "positions": np.mod(q + displacement, MESH_BOX_LENGTH),
        "velocities": -displacement,
        "cells": MESH_CELLS,
        "length": MESH_BOX_LENGTH,
        "time_step": MESH_TIME_STEP,
        "steps": MESH_STEPS,
    }
