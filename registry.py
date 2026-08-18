"""Declarative 22-row paper ledger plus the DF4 extension."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import clusters, empirical, identities, validators
from .orchestrator import PredictionOrchestrator
from .result import Result
from .validators import ValidationMetadata

Adapter = Callable[[Path], Any]
ModelMethod = Callable[[PredictionOrchestrator, Any], Any]
BoundModelMethod = Callable[[Any], Any]
Validator = Callable[[Any, ValidationMetadata], Result]


@dataclass(frozen=True)
class BoundCase:
    adapter: Adapter
    model_method: BoundModelMethod
    validator: Validator
    metadata: ValidationMetadata

    def run(self, data_dir: Path) -> Result:
        return execute_pipeline(
            self.adapter, self.model_method, self.validator, self.metadata, data_dir
        )


@dataclass(frozen=True)
class CaseSpec:
    adapter: Adapter
    model_method: ModelMethod
    validator: Validator
    metadata: ValidationMetadata
    absolute_tolerance: float

    def bind(self, model: PredictionOrchestrator) -> BoundCase:
        return BoundCase(
            self.adapter, self.model_method.__get__(model, type(model)),
            self.validator, self.metadata,
        )


def execute_pipeline(
    adapter: Adapter, model_method: BoundModelMethod, validator: Validator,
    metadata: ValidationMetadata, data_dir: Path,
) -> Result:
    return validator(model_method(adapter(data_dir)), metadata)


def _case(
    name: str, adapter: Adapter, model_method: ModelMethod, validator: Validator,
    evidence: str, method: str, metric: str, expected: float, tolerance: float,
) -> tuple[str, CaseSpec]:
    metadata = ValidationMetadata(name, evidence, method, metric, expected)
    return name, CaseSpec(adapter, model_method, validator, metadata, tolerance)


FORWARD = "released_data_forward_calculation"
EQUIVALENT = "released_data_model_equivalent"
PROXY = "released_data_proxy_map_calculation"
PUBLISHED = "embedded_published_summary_calculation"
IDENTITY = "equation_identity"
SYNTHETIC = "synthetic_numerical_check"
CLUSTER_METHOD = "whole-cluster nested scale selection and nonnegative projection"

CASES = dict((
    _case("galaxy_dynamics", empirical.galaxy_dynamics,
          PredictionOrchestrator.predict_galaxy_dynamics, validators.validate_galaxy,
          FORWARD, "SPARC radial rows; fixed mass-to-light ratios; no fitted galaxy terms",
          "velocity_rmse_km_s", 22.75647592907974, 1e-10),
    _case("planck_spectra", empirical.planck_spectra,
          PredictionOrchestrator.predict_planck_spectra, validators.validate_planck_spectra,
          EQUIVALENT, "CAMB pressureless endpoint and released Plik-lite covariance",
          "chi2_per_bin", 0.9561900562931057, 2e-8),
    _case("planck_lensing", empirical.planck_lensing,
          PredictionOrchestrator.predict_planck_lensing, validators.validate_planck_lensing,
          EQUIVALENT, "CAMB Weyl spectrum and released Planck lensing windows/covariance",
          "chi2_per_band", 1.0044317713248767, 2e-8),
    _case("sdss_eboss", empirical.sdss_eboss,
          PredictionOrchestrator.predict_sdss_endpoint, validators.validate_sdss,
          EQUIVALENT, "published vectors evaluated with released covariance matrices",
          "chi2_per_point", 0.9296973049506106, 1e-10),
    _case("bullet_centroid", identities.bullet_centroid,
          PredictionOrchestrator.predict_bullet_centroid, validators.validate_scalar,
          PUBLISHED, "two-component centroid using published 15% stellar fraction",
          "predicted_centroid_kpc", 86.63927517909818, 1e-12),
    _case("bullet_aperture_peaks", identities.bullet_aperture_peaks,
          PredictionOrchestrator.predict_bullet_apertures,
          validators.validate_bullet_apertures, PUBLISHED,
          "Clowe aperture masses with BCG-anchored nuisance Monte Carlo",
          "main_predicted_plasma_kappa", 0.03259564149552568, 1e-12),
    _case("held_cluster_map_prediction", clusters.cluster_maps,
          PredictionOrchestrator.predict_cluster_proxies, validators.validate_cluster,
          PROXY, CLUSTER_METHOD, "mean_held_correlation", 0.733956807712196, 1e-10),
    _case("hff_lens_robustness", clusters.cluster_maps,
          PredictionOrchestrator.predict_cluster_proxies, validators.validate_cluster,
          PROXY, CLUSTER_METHOD, "minimum_hff_nested_correlation",
          0.6532703491173005, 1e-10),
    _case("abell520_morphology", clusters.cluster_maps,
          PredictionOrchestrator.predict_cluster_proxies, validators.validate_cluster,
          PROXY, CLUSTER_METHOD, "minimum_abell520_nested_correlation",
          0.4811535937624367, 1e-10),
    _case("retrained_cluster_closure", clusters.cluster_maps,
          PredictionOrchestrator.predict_cluster_proxies, validators.validate_cluster,
          PROXY, CLUSTER_METHOD, "mean_held_correlation", 0.733956807712196, 1e-10),
    _case("jades", empirical.jades, PredictionOrchestrator.predict_jades_abundance,
          validators.validate_scalar, EQUIVALENT,
          "catalog target check plus CAMB/Sheth-Tormen abundance",
          "probability_at_least_one", 0.25861854433715203, 2e-8),
    _case("lensing_consistency", identities.lensing_consistency,
          PredictionOrchestrator.predict_lensing_identity, validators.validate_scalar,
          IDENTITY, "equal scalar potentials are imposed by the model construction",
          "phi_over_psi", 1.0, 0.0),
    _case("cmb_isocurvature", identities.cmb_isocurvature,
          PredictionOrchestrator.predict_isocurvature_identity, validators.validate_scalar,
          IDENTITY, "source-fixed adiabatic initialization", "response_fraction", 0.0, 0.0),
    _case("bbn_background", identities.bbn_background,
          PredictionOrchestrator.predict_bbn_background, validators.validate_scalar,
          PUBLISHED, "matter-to-radiation scaling at z=1e9",
          "fractional_H_change", 1.4390477606252498e-6, 1e-18),
    _case("ppn", identities.ppn, PredictionOrchestrator.predict_ppn_identity,
          validators.validate_scalar, IDENTITY,
          "vacuum Einstein kinetic and nonlinear operators are unchanged",
          "gamma_ppn", 1.0, 0.0),
    _case("gw_propagation", identities.gw_propagation,
          PredictionOrchestrator.predict_gw_identity, validators.validate_scalar,
          IDENTITY, "unchanged massless tensor kinetic operator",
          "tensor_speed_minus_c_over_c", 0.0, 0.0),
    _case("binary_orbital_decay", identities.binary_orbital_decay,
          PredictionOrchestrator.predict_binary_identity, validators.validate_scalar,
          IDENTITY, "no additional radiative pole in the specified completion",
          "Pdot_over_GR", 1.0, 0.0),
    _case("sgr_a", identities.sgr_a, PredictionOrchestrator.predict_shadow_scale,
          validators.validate_scalar, PUBLISHED,
          "Schwarzschild shadow scale from published mass and distance",
          "bare_shadow_diameter_microarcsec", 53.255275180974486, 1e-10),
    _case("bianchi_transversality", identities.bianchi_transversality,
          PredictionOrchestrator.predict_bianchi_response, validators.validate_bianchi,
          SYNTHETIC, "1000 deterministic Lorentzian spin-2 plus scalar projections",
          "maximum_relative_divergence", 8.783193835147004e-16, 1e-25),
    _case("gate_robustness", identities.gate_robustness,
          PredictionOrchestrator.predict_gate_response, validators.validate_gate,
          SYNTHETIC, "fixed exhaustive 241 by 221 parameter grid",
          "valid_grid_points", 4732.0, 0.0),
    _case("linear_stability", identities.linear_stability,
          PredictionOrchestrator.predict_linear_stability, validators.validate_stability,
          SYNTHETIC, "positivity, bounded-gate, pole, and characteristic audit",
          "all_flags_pass", 1.0, 0.0),
    _case("nonlinear_pm_equivalence", identities.nonlinear_pm_equivalence,
          PredictionOrchestrator.predict_particle_mesh, validators.validate_particle_mesh,
          SYNTHETIC, "repeated two-component particle-mesh evolution",
          "maximum_density_difference", 0.0, 1e-10),
    _case("df4", empirical.df4, PredictionOrchestrator.predict_df4,
          validators.validate_df4, FORWARD,
          "pinned PDF tracers with prior-predictive nuisance marginalization",
          "marginal_delta_deviance", 1.2660550030192468, 1e-12),
))

VALIDATION_CASES = tuple(CASES)
LEDGER_CASES = (
    "galaxy_dynamics", "planck_spectra", "planck_lensing", "sdss_eboss",
    "bullet_centroid", "bullet_aperture_peaks", "held_cluster_map_prediction",
    "hff_lens_robustness", "abell520_morphology", "retrained_cluster_closure",
    "jades", "lensing_consistency", "cmb_isocurvature", "bbn_background",
    "ppn", "gw_propagation", "binary_orbital_decay", "sgr_a",
    "bianchi_transversality", "gate_robustness", "linear_stability",
    "nonlinear_pm_equivalence",
)
REFERENCES = {
    name: (spec.metadata.expected_value, spec.absolute_tolerance)
    for name, spec in CASES.items()
}

if VALIDATION_CASES != (*LEDGER_CASES, "df4"):
    raise RuntimeError("registry must contain the fixed 22-row ledger plus DF4")
