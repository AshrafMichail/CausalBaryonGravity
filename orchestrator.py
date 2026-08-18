"""Short applications of the explicit validation models."""

from __future__ import annotations

from dataclasses import dataclass, field

from .benchmarks import BaryonsOnlyModel, MONDModel, NFWModel
from .model import CausalBaryonModel
from .particle_mesh import ReferenceParticleMeshModel
from .records import Record
from .released import ReleasedEndpointModel


@dataclass(frozen=True)
class PredictionOrchestrator:
    """Own one model of each kind and compose only named comparisons."""

    causal_model: CausalBaryonModel = field(default_factory=CausalBaryonModel)
    mond_model: MONDModel = field(default_factory=MONDModel)
    nfw_model: NFWModel = field(default_factory=NFWModel)
    baryons_only_model: BaryonsOnlyModel = field(default_factory=BaryonsOnlyModel)
    reference_particle_mesh_model: ReferenceParticleMeshModel = field(
        default_factory=ReferenceParticleMeshModel
    )
    released_endpoint_model: ReleasedEndpointModel = field(
        default_factory=ReleasedEndpointModel
    )

    def parameters(self) -> dict[str, float | list[float]]:
        return self.causal_model.parameters()

    def predict_galaxy_dynamics(self, inputs: Record) -> Record:
        baseline = self.baryons_only_model.predict_galaxy(inputs)
        response = self.causal_model.predict_galaxy_response(
            baseline, inputs["external_fields"]
        )
        return {
            "observed_speed_km_s": baseline["observed_speed_km_s"],
            "model_speeds_km_s": {
                "Causal Baryon": response["speed_km_s"],
                "Simple MOND": self.mond_model.predict_galaxy_speed(
                    baseline["baryonic_acceleration_m_s2"], baseline["radius_kpc"]
                ),
                "Baryons": baseline["speed_km_s"],
            },
            "galaxies": baseline["galaxies"],
            "external_acceleration_m_s2": response["external_acceleration_m_s2"],
            "sources": baseline["sources"],
        }

    def predict_planck_spectra(self, inputs: Record) -> Record:
        return self.causal_model.predict_planck_spectra(inputs)

    def predict_planck_lensing(self, inputs: Record) -> Record:
        return self.causal_model.predict_planck_lensing(inputs)

    def predict_sdss_endpoint(self, inputs: Record) -> Record:
        return self.released_endpoint_model.predict_sdss(inputs)

    def predict_bullet_centroid(self, inputs: Record) -> Record:
        return self.causal_model.predict_bullet_centroid(inputs)

    def predict_bullet_apertures(self, inputs: Record) -> Record:
        return self.causal_model.predict_bullet_apertures(inputs)

    def predict_cluster_proxies(self, inputs: Record) -> Record:
        return self.causal_model.predict_cluster_proxies(inputs)

    def predict_jades_abundance(self, inputs: Record) -> Record:
        return self.causal_model.predict_jades_abundance(inputs)

    def predict_lensing_identity(self, inputs: Record) -> Record:
        return self.causal_model.predict_lensing_identity(inputs)

    def predict_isocurvature_identity(self, inputs: Record) -> Record:
        return self.causal_model.predict_isocurvature_identity(inputs)

    def predict_bbn_background(self, inputs: Record) -> Record:
        return self.causal_model.predict_bbn_background(inputs)

    def predict_ppn_identity(self, inputs: Record) -> Record:
        return self.causal_model.predict_ppn_identity(inputs)

    def predict_gw_identity(self, inputs: Record) -> Record:
        return self.causal_model.predict_gw_identity(inputs)

    def predict_binary_identity(self, inputs: Record) -> Record:
        return self.causal_model.predict_binary_identity(inputs)

    def predict_shadow_scale(self, inputs: Record) -> Record:
        return self.causal_model.predict_shadow_scale(inputs)

    def predict_bianchi_response(self, inputs: Record) -> Record:
        return self.causal_model.predict_bianchi_response(inputs)

    def predict_gate_response(self, inputs: Record) -> Record:
        return self.causal_model.predict_gate_response(inputs)

    def predict_linear_stability(self, inputs: Record) -> Record:
        return self.causal_model.predict_linear_stability(inputs)

    def predict_particle_mesh(self, inputs: Record) -> Record:
        ratio = self.causal_model.response_to_baryon_ratio()
        return {
            "densities": {
                "Reference": self.reference_particle_mesh_model.predict_density(
                    inputs, ratio
                ),
                "Causal Baryon": self.causal_model.predict_particle_mesh_density(inputs),
            },
            "particles_per_component": len(inputs["positions"]),
            "mesh_cells": inputs["cells"],
            "steps": inputs["steps"],
        }

    def predict_df4(self, inputs: Record) -> Record:
        baseline, generator = self.baryons_only_model.predict_df4(inputs)
        state = baseline["state"]
        dispersions = {
            "Baryons": state["baryonic_dispersion_km_s"],
            "Causal Baryon Model": self.causal_model.predict_df4_response(state),
            **self.nfw_model.predict_df4_dispersions(state, generator),
        }
        return {
            "sigma_samples_km_s": dispersions,
            "velocities_km_s": baseline["velocities_km_s"],
            "errors_km_s": baseline["errors_km_s"],
            "tracer_count": baseline["tracer_count"],
            "samples": baseline["samples"],
            "random_seed": baseline["random_seed"],
            "source": baseline["source"],
        }
