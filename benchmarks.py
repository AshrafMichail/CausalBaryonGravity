"""Explicit alternative models used in named validation comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import NewtonianDynamics
from .records import Record


@dataclass(frozen=True)
class MONDModel(NewtonianDynamics):
    """Fixed simple-interpolation MOND model."""

    acceleration_scale_m_s2: float = 1.2e-10
    # nu(y) = offset + sqrt(offset**2 + 1 / y) is the simple interpolation.
    _simple_interpolation_offset = 0.5
    _minimum_acceleration_ratio = 1e-30

    def acceleration(self, baryonic_acceleration_m_s2: np.ndarray | float) -> np.ndarray:
        """Apply the fixed simple-interpolation MOND acceleration law."""
        baryonic = np.asarray(baryonic_acceleration_m_s2, dtype=float)
        if np.any(baryonic < 0.0):
            raise ValueError("baryonic acceleration must be nonnegative")
        ratio = np.maximum(
            baryonic / self.acceleration_scale_m_s2, self._minimum_acceleration_ratio
        )
        offset = self._simple_interpolation_offset
        return baryonic * (offset + np.sqrt(offset**2 + 1.0 / ratio))

    def predict_galaxy_speed(
        self,
        baryonic_acceleration_m_s2: np.ndarray,
        radius_kpc: np.ndarray,
    ) -> np.ndarray:
        """Convert the MOND acceleration prediction into rotation speed."""
        return self.speed_from_acceleration(
            self.acceleration(baryonic_acceleration_m_s2), radius_kpc
        )


@dataclass(frozen=True)
class NFWModel(NewtonianDynamics):
    """NFW halo profile with the two explicit DF4 prior families."""

    reference_density_msun_kpc3: float = 127.0
    overdensity: float = 200.0
    canonical_mass_msun: float = 7.5e10
    canonical_mass_log10_sigma: float = 0.30
    canonical_concentration: float = 10.0
    canonical_concentration_log10_sigma: float = 0.12
    flexible_mass_log10_minimum: float = 7.0
    flexible_mass_log10_maximum: float = 11.5
    flexible_concentration: float = 10.0
    flexible_concentration_log10_sigma: float = 0.20

    def profile(self, scaled_radius: np.ndarray) -> np.ndarray:
        """Evaluate the dimensionless enclosed-mass factor of an NFW halo."""
        return np.log1p(scaled_radius) - scaled_radius / (1.0 + scaled_radius)

    def enclosed_mass(
        self,
        radius_kpc: np.ndarray,
        mass_msun: np.ndarray,
        concentration: np.ndarray,
    ) -> np.ndarray:
        """Compute NFW mass enclosed by each tracer radius."""
        r200 = (
            3.0 * mass_msun
            / (4.0 * np.pi * self.overdensity * self.reference_density_msun_kpc3)
        ) ** (1.0 / 3.0)
        scaled_radius = radius_kpc / (r200 / concentration)
        return mass_msun * self.profile(scaled_radius) / self.profile(concentration)

    def compact_dispersion(
        self,
        state: Record,
        halo_mass_msun: np.ndarray,
        concentration: np.ndarray,
    ) -> np.ndarray:
        """Add an NFW halo to the stellar mass before estimating dispersion."""
        return state["structure"] * self.baryonic_dispersion(
            state["stellar_enclosed_mass_msun"]
            + self.enclosed_mass(
                state["radius_kpc"],
                halo_mass_msun,
                concentration,
            ),
            state["radius_kpc"],
        )

    def predict_df4_dispersions(
        self,
        state: Record,
        generator: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """Draw canonical and flexible NFW prior-predictive dispersions."""
        count = len(state["radius_kpc"])
        canonical_mass = 10 ** generator.normal(
            np.log10(self.canonical_mass_msun), self.canonical_mass_log10_sigma, count
        )
        canonical_concentration = 10 ** generator.normal(
            np.log10(self.canonical_concentration),
            self.canonical_concentration_log10_sigma, count,
        )
        canonical_sigma = self.compact_dispersion(
            state, canonical_mass, canonical_concentration
        )
        flexible_mass = 10 ** generator.uniform(
            self.flexible_mass_log10_minimum, self.flexible_mass_log10_maximum, count
        )
        flexible_concentration = 10 ** generator.normal(
            np.log10(self.flexible_concentration),
            self.flexible_concentration_log10_sigma, count,
        )
        flexible_sigma = self.compact_dispersion(
            state, flexible_mass, flexible_concentration
        )
        return {"Canonical NFW": canonical_sigma, "Flexible NFW": flexible_sigma}

@dataclass(frozen=True)
class BaryonsOnlyModel(NewtonianDynamics):
    """Newtonian baryonic baselines for galaxy and DF4 comparisons."""

    disk_mass_to_light: float = 0.5
    bulge_mass_to_light: float = 0.7
    df4_stellar_mass_msun: float = 1.5e8
    df4_stellar_mass_log10_sigma: float = 0.20
    df4_effective_radius_kpc: float = 1.6
    df4_effective_radius_sigma_kpc: float = 0.2
    df4_effective_radius_bounds_kpc: tuple[float, float] = (0.8, 2.5)
    df4_host_speed_km_s: float = 210.0
    df4_host_speed_sigma_km_s: float = 30.0
    df4_host_speed_bounds_km_s: tuple[float, float] = (100.0, 320.0)
    df4_host_distance_kpc: float = 80.0
    df4_host_distance_sigma_kpc: float = 20.0
    df4_host_distance_bounds_kpc: tuple[float, float] = (35.0, 150.0)
    df4_structure_log10_sigma: float = 0.08

    def predict_galaxy(self, inputs: Record) -> Record:
        """Construct the fixed mass-to-light baryonic SPARC baseline."""
        baryonic_speed_squared = (
            np.sign(inputs["gas_speed_km_s"]) * inputs["gas_speed_km_s"] ** 2
            + self.disk_mass_to_light * inputs["disk_speed_km_s"] ** 2
            + self.bulge_mass_to_light * inputs["bulge_speed_km_s"] ** 2
        )
        selected = baryonic_speed_squared > 0.0
        speed = np.sqrt(baryonic_speed_squared[selected])
        radius = inputs["radius_kpc"][selected]
        return {
            "observed_speed_km_s": inputs["observed_speed_km_s"][selected],
            "speed_km_s": speed,
            "baryonic_acceleration_m_s2": self.acceleration_from_speed(speed, radius),
            "radius_kpc": radius,
            "galaxies": inputs["galaxies"][selected],
            "sources": inputs["sources"],
        }

    def predict_df4(self, inputs: Record) -> tuple[Record, np.random.Generator]:
        """Draw the shared DF4 baryonic and environmental nuisance state."""
        generator = np.random.default_rng(inputs["random_seed"])
        samples = inputs["samples"]
        stellar_mass = 10 ** generator.normal(
            np.log10(self.df4_stellar_mass_msun), self.df4_stellar_mass_log10_sigma,
            samples,
        )
        effective_radius = np.clip(
            generator.normal(
                self.df4_effective_radius_kpc, self.df4_effective_radius_sigma_kpc,
                samples,
            ),
            *self.df4_effective_radius_bounds_kpc,
        )
        host_speed = np.clip(
            generator.normal(
                self.df4_host_speed_km_s, self.df4_host_speed_sigma_km_s, samples
            ),
            *self.df4_host_speed_bounds_km_s,
        )
        host_distance = np.clip(
            generator.normal(
                self.df4_host_distance_kpc, self.df4_host_distance_sigma_kpc, samples
            ),
            *self.df4_host_distance_bounds_kpc,
        )
        structure = 10 ** generator.normal(0.0, self.df4_structure_log10_sigma, samples)
        radius = self.compact_radius(effective_radius)
        stellar_enclosed = self.compact_enclosed_mass(stellar_mass)
        state = {
            "radius_kpc": radius,
            "stellar_enclosed_mass_msun": stellar_enclosed,
            "baryonic_acceleration_m_s2": self.point_mass_acceleration(
                stellar_enclosed, radius
            ),
            "external_acceleration_m_s2": self.acceleration_from_speed(
                host_speed, host_distance
            ),
            "baryonic_dispersion_km_s": (
                structure * self.baryonic_dispersion(stellar_enclosed, radius)
            ),
            "structure": structure,
        }
        errors = 0.5 * (inputs["upper_errors_km_s"] + inputs["lower_errors_km_s"])
        return (
            {
                "state": state,
                "velocities_km_s": inputs["velocities_km_s"],
                "errors_km_s": errors,
                "tracer_count": len(inputs["tracer_ids"]),
                "samples": samples,
                "random_seed": inputs["random_seed"],
                "source": inputs["source"],
            },
            generator,
        )
