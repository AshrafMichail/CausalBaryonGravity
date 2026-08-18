"""Causal Baryon equations and forward predictions."""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import camb
import numpy as np
from astropy.cosmology import FlatLambdaCDM, Planck18
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter
from scipy.optimize import nnls

from .dynamics import NewtonianDynamics
from .particle_mesh import ParticleMeshEngine
from .records import (
    BULLET_BARYON_MASS_COLUMNS, BULLET_BCG_POSITION, BULLET_CONVERGENCE_COLUMNS,
    BULLET_PLASMA_POSITION, Record,
)

# RELICS Abell 520 member catalogue: zero-based columns and membership cuts.
A520_RA_COLUMN, A520_DEC_COLUMN = 1, 2
A520_STELLARITY_COLUMN = 7
A520_MAGNITUDE_COLUMN = 26
A520_FLUX_NJY_COLUMN = 30
A520_REDSHIFT_COLUMN, A520_REDSHIFT_ODDS_COLUMN = 63, 67
A520_MEMBER_REDSHIFT_RANGE = (0.12, 0.28)
A520_MINIMUM_REDSHIFT_ODDS = 0.5
A520_MAXIMUM_STELLARITY = 0.8
A520_MAXIMUM_MAGNITUDE = 26.0


@dataclass(frozen=True)
class CausalBaryonModel(NewtonianDynamics):
    """Causal response equations, identities, and forward calculations."""

    a0_m_s2: float = 1.45947e-10
    alpha: float = 0.898563
    external_acceleration_m_s2: float = 1.72218e-12
    hubble_km_s_mpc: float = 67.32
    omega_b_h2: float = 0.02238
    response_h2: float = 0.1200
    neutrino_mass_ev: float = 0.06
    optical_depth: float = 0.0543
    scalar_amplitude: float = 2.10e-9
    scalar_index: float = 0.9660
    gate_threshold: float = 3e-6
    gate_power: float = 8.76572
    cluster_smoothing_scales_kpc: tuple[float, ...] = (50.0, 100.0, 150.0, 200.0)
    # Physical constants and unit conversions.
    _speed_of_light_m_s, _solar_mass_kg = 299792458.0, 1.98847e30
    _parsec_m, _gravitational_constant_si = 3.085677581491367e16, 6.67430e-11
    _meters_per_kilometer, _microarcsec_per_radian = 1000.0, 206265.0e6
    _hubble_reduction_km_s_mpc, _cmb_temperature_k = 100.0, 2.7255
    _neutrino_mass_per_omega_h2_ev = 93.14
    _critical_density_msun_h2_mpc3 = 2.77536627e11
    _arcsec_per_arcmin, _arcmin2_per_degree2 = 60.0, 3600.0
    _full_sky_square_degrees = 4.0 * np.pi * (180.0 / np.pi) ** 2
    # Floors and caps that keep the response and the fitted maps finite.
    _minimum_variance = 1e-30
    _minimum_positive_acceleration_m_s2 = 1e-300
    _response_quarter = 0.25
    _minimum_map_scale, _null_wavevector_tolerance = 1e-12, 1e-12
    # Released-product configuration.
    _chae_fitted_log10_cutoff = -10.6
    _massive_neutrino_count = 1
    _cmb_lmax, _cmb_lens_potential_accuracy = 2600, 0
    _lensing_lmax, _lensing_potential_accuracy = 2500, 1
    _lensing_spectrum_column, _plik_first_multipole = 0, 30
    _plik_spectrum_columns = (0, 3, 1)  # CAMB total TT, TE, EE columns
    # Published identity bounds and geometric conversions.
    _planck_isocurvature_upper_bound = 0.038
    _general_relativity_gamma, _general_relativity_beta = 1.0, 1.0
    _schwarzschild_shadow_diameter_factor = 6.0 * np.sqrt(3.0)
    _spacetime_dimensions, _bianchi_timelike_offset = 4, 3.0
    # JADES abundance settings; grids are (start, stop, count) arguments.
    _matter_power_kmax_mpc, _matter_power_redshift_margin = 100.0, 0.5
    _sigma_wavenumber_grid_mpc = (1e-4, 100.0, 5000)
    _halo_mass_grid_msun = (1e8, 1e13, 500)
    _star_formation_efficiency_grid = (0.01, 1.0, 160)
    _top_hat_small_argument, _spherical_collapse_delta_c = 1e-4, 1.686
    _sheth_tormen_normalization, _sheth_tormen_a, _sheth_tormen_p = 0.3222, 0.707, 0.3
    # Cluster map preprocessing; percentiles are (background, scale) pairs.
    _cluster_map_kinds = ("light", "gas")
    _map_smoothing_arcsec, _hff_footprint_threshold = 12.0, 0.5
    _hff_map_percentiles = (15.0, 90.0)
    _abell520_map_percentiles = (10.0, 95.0)
    _abell520_weight_percentile_cap, _abell520_light_support_percentile = 95, 1
    _abell520_support_smoothing_pixels, _abell520_minimum_support_weight = 20.0, 0.01
    _particle_mesh_engine: ParticleMeshEngine = field(
        default_factory=ParticleMeshEngine, init=False, repr=False, compare=False
    )

    def parameters(self) -> dict[str, float | list[float]]:
        """Report every declared model parameter, in declaration order."""
        values = {
            item.name: getattr(self, item.name) for item in fields(self) if item.init
        }
        values["cluster_smoothing_scales_kpc"] = list(self.cluster_smoothing_scales_kpc)
        return values

    def response_to_baryon_ratio(self) -> float:
        return self.response_h2 / self.omega_b_h2

    def _algebraic_external_response_nu(
        self,
        internal_newtonian_acceleration: np.ndarray | float,
        external_newtonian_acceleration: np.ndarray | float,
    ) -> np.ndarray:
        """Return the current algebraic response with external screening."""
        internal, external = np.broadcast_arrays(
            np.asarray(internal_newtonian_acceleration, dtype=float),
            np.asarray(external_newtonian_acceleration, dtype=float),
        )
        if np.any(internal < 0.0) or np.any(external < 0.0):
            raise ValueError("accelerations must be nonnegative")
        if (
            self.a0_m_s2 <= 0.0
            or self.external_acceleration_m_s2 <= 0.0
            or self.gate_power <= 0.0
        ):
            raise ValueError("transition scales must be positive")
        isolated_response = (
            np.sqrt(
                self._response_quarter
                + self.a0_m_s2
                / np.maximum(
                    internal,
                    self._minimum_positive_acceleration_m_s2,
                )
            )
            - 0.5
        )
        response_weight = 1.0 / (
            1.0
            + np.power(
                external / self.external_acceleration_m_s2,
                self.gate_power,
            )
        )
        return 1.0 + response_weight * isolated_response

    def _infer_newtonian_acceleration(
        self,
        dynamic_acceleration: np.ndarray | float,
    ) -> np.ndarray:
        """Invert the isolated response to recover the Newtonian host field."""
        dynamic = np.asarray(dynamic_acceleration, dtype=float)
        if np.any(dynamic < 0.0):
            raise ValueError("dynamic_acceleration must be nonnegative")
        if self.a0_m_s2 <= 0.0 or self.alpha < 0.0:
            raise ValueError("response scales must be nonnegative")
        dynamic_ratio = dynamic / self.a0_m_s2
        discriminant = np.sqrt(
            np.square(dynamic_ratio - self.alpha)
            + 4.0 * dynamic_ratio
        )
        response = np.divide(
            2.0,
            discriminant + dynamic_ratio - self.alpha,
            out=np.full_like(dynamic_ratio, np.inf),
            where=dynamic_ratio > 0.0,
        )
        return np.divide(
            self.a0_m_s2,
            response * (response + 1.0),
            out=np.zeros_like(dynamic_ratio),
            where=np.isfinite(response),
        )

    def susceptibility(
        self, local_acceleration_m_s2: np.ndarray | float,
        external_acceleration_m_s2: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """Evaluate the causal response strength with external screening."""
        local, external = np.broadcast_arrays(
            np.asarray(local_acceleration_m_s2, dtype=float),
            np.asarray(external_acceleration_m_s2, dtype=float))
        if np.any(local < 0.0) or np.any(external < 0.0):
            raise ValueError("accelerations must be nonnegative")
        response = self._algebraic_external_response_nu(local, external)
        return self.alpha * (response - 1.0)

    def galaxy_acceleration(
        self, baryonic_acceleration: np.ndarray | float,
        external_acceleration: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """Add the screened causal response to the baryonic acceleration."""
        baryonic = np.asarray(baryonic_acceleration, dtype=float)
        return baryonic * (
            1.0 + self.susceptibility(baryonic, external_acceleration)
        )

    def response_rescaled_dispersion(
        self, baryonic_dispersion_km_s: np.ndarray | float,
        baryonic_acceleration_m_s2: np.ndarray | float,
        external_acceleration_m_s2: np.ndarray | float,
    ) -> np.ndarray:
        """Rescale a baryonic dispersion by the causal acceleration ratio."""
        baryonic_acceleration = np.asarray(baryonic_acceleration_m_s2)
        return np.asarray(baryonic_dispersion_km_s) * np.sqrt(
            self.galaxy_acceleration(baryonic_acceleration, external_acceleration_m_s2)
            / baryonic_acceleration
        )

    def compact_dispersion(
        self, stellar_mass_msun: float, effective_radius_kpc: float,
        external_acceleration_m_s2: float,
    ) -> float:
        """Apply the complete compact-system response from observable inputs."""
        radius = self.compact_radius(effective_radius_kpc)
        enclosed_mass = self.compact_enclosed_mass(stellar_mass_msun)
        baryonic_acceleration = self.point_mass_acceleration(enclosed_mass, radius)
        return float(
            self.response_rescaled_dispersion(
                self.baryonic_dispersion(enclosed_mass, radius),
                baryonic_acceleration,
                external_acceleration_m_s2,
            )
        )

    def predict_galaxy_response(self, baseline: Record, external_fields: Record) -> Record:
        """Match galaxy environments and apply the causal rotation-curve law."""
        baryonic_acceleration = baseline["baryonic_acceleration_m_s2"]
        selected_external = (
            external_fields["fitted_log10_acceleration"]
            < self._chae_fitted_log10_cutoff
        )
        external_by_galaxy = dict(
            zip(
                external_fields["galaxies"][selected_external],
                external_fields["environmental_log10_acceleration"][selected_external],
            )
        )
        external = np.asarray(
            [
                10.0 ** external_by_galaxy[galaxy] * self.a0_m_s2
                if galaxy in external_by_galaxy
                else 0.0
                for galaxy in baseline["galaxies"]
            ],
            dtype=float,
        )
        return {
            "speed_km_s": self.speed_from_acceleration(
                self.galaxy_acceleration(baryonic_acceleration, external),
                baseline["radius_kpc"],
            ),
            "external_acceleration_m_s2": external,
        }

    def reduced_hubble(self) -> float:
        return self.hubble_km_s_mpc / self._hubble_reduction_km_s_mpc

    def cosmology_state(self) -> dict[str, float | int]:
        return {
            "H0": self.hubble_km_s_mpc,
            "ombh2": self.omega_b_h2,
            "omch2": self.response_h2,
            "omk": 0.0,
            "mnu": self.neutrino_mass_ev,
            "num_massive_neutrinos": self._massive_neutrino_count,
            "tau": self.optical_depth,
        }

    def primordial_power(self) -> dict[str, float]:
        return {"As": self.scalar_amplitude, "ns": self.scalar_index}

    def build_camb_parameters(
        self, *, lmax: int | None = None, lens_potential_accuracy: int = 0,
        matter_redshifts: list[float] | None = None, matter_kmax: float | None = None,
    ) -> camb.CAMBparams:
        """Build one CAMB configuration from the model's cosmological endpoint."""
        parameters = camb.CAMBparams()
        parameters.set_cosmology(**self.cosmology_state())
        parameters.InitPower.set_params(**self.primordial_power())
        if lmax is not None:
            parameters.set_for_lmax(lmax, lens_potential_accuracy=lens_potential_accuracy)
        if matter_redshifts is not None and matter_kmax is not None:
            parameters.set_matter_power(redshifts=matter_redshifts, kmax=matter_kmax)
        return parameters

    def _bin_cmb_spectrum(
        self, values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray,
        weights: np.ndarray, count: int,
    ) -> np.ndarray:
        result = np.empty(count)
        first = self._plik_first_multipole
        for index in range(count):
            lo, hi = minimum[index], maximum[index]
            result[index] = np.sum(
                values[lo + first : hi + first + 1] * weights[lo : hi + 1]
            )
        return result

    def predict_planck_spectra(self, inputs: Record) -> Record:
        """Run CAMB and bin its CMB spectra through the released Planck windows."""
        parameters = self.build_camb_parameters(
            lmax=self._cmb_lmax,
            lens_potential_accuracy=self._cmb_lens_potential_accuracy,
        )
        spectra = camb.get_results(parameters).get_cmb_power_spectra(
            parameters, CMB_unit="muK", raw_cl=False
        )["total"]
        ell = np.arange(len(spectra), dtype=float)
        factor = ell * (ell + 1.0) / (2.0 * np.pi)
        factor[:2] = 1.0
        blocks = tuple(
            self._bin_cmb_spectrum(
                spectra[:, column] / factor, inputs["bin_minimum"],
                inputs["bin_maximum"], inputs["bin_weights"], count,
            )
            for column, count in zip(self._plik_spectrum_columns, inputs["bin_counts"])
        )
        predicted = np.concatenate(blocks)
        return {
            "observed": inputs["observed"],
            "predicted": predicted,
            "covariance": inputs["covariance"],
            "calibration_sigma": inputs["calibration_sigma"],
            "source": inputs["source"],
        }

    def _apply_spectrum_windows(
        self, spectrum: np.ndarray, windows: tuple[Record, ...]
    ) -> np.ndarray:
        rows = [
            np.sum(spectrum[window["multipoles"]] * window["weights"])
            for window in windows
        ]
        return np.asarray(rows, dtype=float)

    def predict_planck_lensing(self, inputs: Record) -> Record:
        """Run CAMB lensing and apply the released Planck bandpower windows."""
        parameters = self.build_camb_parameters(
            lmax=self._lensing_lmax,
            lens_potential_accuracy=self._lensing_potential_accuracy,
        )
        spectrum = camb.get_results(parameters).get_lens_potential_cls(
            lmax=self._lensing_lmax, raw_cl=False
        )[:, self._lensing_spectrum_column]
        predicted = (
            self._apply_spectrum_windows(spectrum, inputs["primary_windows"])
            + self._apply_spectrum_windows(spectrum, inputs["correction_windows"])
            - inputs["fiducial_correction"]
        )
        return {
            "observed": inputs["observed"],
            "predicted": predicted,
            "covariance": inputs["covariance"],
            "source": inputs["source"],
        }

    def predict_bullet_centroid(self, inputs: Record) -> Record:
        ratio = self.response_to_baryon_ratio()
        value = (
            inputs["stellar_fraction"] * inputs["stellar_position_kpc"]
            + (1.0 - inputs["stellar_fraction"]) * inputs["plasma_position_kpc"]
            + ratio * inputs["stellar_position_kpc"]
        ) / (1.0 + ratio)
        return {
            "value": float(value),
            "details": {"collisionless_to_baryon_ratio": ratio},
        }

    def predict_bullet_apertures(self, inputs: Record) -> Record:
        """Propagate published Bullet aperture uncertainties through the response."""
        rng = np.random.default_rng(inputs["random_seed"])
        response = self.response_to_baryon_ratio()
        samples = inputs["samples"]
        convergence, convergence_sigma = BULLET_CONVERGENCE_COLUMNS
        predictions = []
        ratios = []
        observed = []
        errors = []
        for system_index in range(len(inputs["system_names"])):
            baryon = []
            for position_index in range(len(inputs["position_names"])):
                row = inputs["measurements"][system_index, position_index]
                baryon.append(sum(
                    np.clip(rng.normal(row[mean], row[sigma], samples), 0.0, None)
                    for mean, sigma in BULLET_BARYON_MASS_COLUMNS
                ))
            effective_bcg = baryon[BULLET_BCG_POSITION] + response * (
                baryon[BULLET_BCG_POSITION] + baryon[BULLET_PLASMA_POSITION]
            )
            ratio = effective_bcg / baryon[BULLET_PLASMA_POSITION]
            bcg = inputs["measurements"][system_index, BULLET_BCG_POSITION]
            plasma = inputs["measurements"][system_index, BULLET_PLASMA_POSITION]
            predictions.append(
                rng.normal(bcg[convergence], bcg[convergence_sigma], samples) / ratio
            )
            ratios.append(ratio)
            observed.append(plasma[convergence])
            errors.append(plasma[convergence_sigma])
        return {
            "system_names": inputs["system_names"],
            "predicted_plasma_kappa": tuple(predictions),
            "bcg_to_plasma_ratio": tuple(ratios),
            "observed_plasma_kappa": np.asarray(observed),
            "plasma_errors": np.asarray(errors),
            "aperture_radius_kpc": inputs["aperture_radius_kpc"],
            "fractional_prediction_error": inputs["fractional_prediction_error"],
            "collisionless_to_baryon_ratio": response,
            "samples": samples,
            "random_seed": inputs["random_seed"],
            "source": inputs["source"],
        }

    def predict_lensing_identity(self, _: Record) -> Record:
        return {"value": 1.0, "details": {}}

    def predict_isocurvature_identity(self, _: Record) -> Record:
        bound = self._planck_isocurvature_upper_bound
        return {"value": 0.0, "details": {"planck_upper_bound": bound}}

    def predict_bbn_background(self, inputs: Record) -> Record:
        """Scale the response density to the Hubble rate change at the BBN epoch."""
        h = self.reduced_hubble()
        omega_response = self.response_h2 / h**2
        ratio = (
            omega_response / inputs["radiation_density_h2"] / (1.0 + inputs["redshift"])
        )
        return {"value": float(0.5 * ratio), "details": {}}

    def predict_ppn_identity(self, _: Record) -> Record:
        """Report the unchanged general-relativity post-Newtonian parameters."""
        gamma = self._general_relativity_gamma
        beta = self._general_relativity_beta
        return {
            "value": gamma,
            "details": {
                "beta_ppn": beta,
                "nordtvedt_eta": 4.0 * beta - gamma - 3.0,
            },
        }

    def predict_gw_identity(self, _: Record) -> Record:
        return {"value": 0.0, "details": {}}

    def predict_binary_identity(self, _: Record) -> Record:
        return {"value": 1.0, "details": {}}

    def predict_shadow_scale(self, inputs: Record) -> Record:
        mass = inputs["mass_msun"] * self._solar_mass_kg
        distance = inputs["distance_pc"] * self._parsec_m
        gravitational_radius = self._gravitational_constant_si * mass / self._speed_of_light_m_s**2
        diameter = (
            self._schwarzschild_shadow_diameter_factor
            * gravitational_radius
            / distance
            * self._microarcsec_per_radian
        )
        return {
            "value": float(diameter),
            "details": {
                "mass_msun": inputs["mass_msun"],
                "distance_pc": inputs["distance_pc"],
                "observed_ring_diameter_microarcsec": (
                    inputs["observed_ring_diameter_microarcsec"]
                ),
                "limitation": "This is a scale check, not an EHT visibility refit.",
            },
        }

    def minkowski_metric(self) -> np.ndarray:
        return np.diag([1.0, -1.0, -1.0, -1.0])

    def lorentzian_response_projector(self, wavevector: np.ndarray) -> np.ndarray:
        minkowski = self.minkowski_metric()
        lowered = minkowski @ wavevector
        norm = float(wavevector @ lowered)
        if abs(norm) < self._null_wavevector_tolerance:
            raise ValueError("projector is singular for a null wavevector")
        theta = minkowski - np.outer(lowered, lowered) / norm
        dimensions = self._spacetime_dimensions
        spin2 = np.empty((dimensions, dimensions, dimensions, dimensions))
        spin0 = np.empty_like(spin2)
        for mu in range(dimensions):
            for nu in range(dimensions):
                for alpha in range(dimensions):
                    for beta in range(dimensions):
                        symmetrized = (
                            theta[mu, alpha] * theta[nu, beta]
                            + theta[mu, beta] * theta[nu, alpha]
                        )
                        # The transverse trace part carries 1 / (D - 1) = 1 / 3.
                        trace = theta[mu, nu] * theta[alpha, beta] / 3.0
                        spin2[mu, nu, alpha, beta] = 0.5 * symmetrized - trace
                        spin0[mu, nu, alpha, beta] = trace
        return spin2 + spin0

    def predict_bianchi_response(self, inputs: Record) -> Record:
        minkowski = self.minkowski_metric()
        rng = np.random.default_rng(inputs["random_seed"])
        samples, dimensions = inputs["samples"], self._spacetime_dimensions
        wavevectors = np.empty((samples, dimensions))
        responses = np.empty((samples, dimensions, dimensions))
        for index in range(samples):
            wave = rng.normal(size=dimensions)
            wave[0] += self._bianchi_timelike_offset
            source = rng.normal(size=(dimensions, dimensions))
            source = 0.5 * (source + source.T)
            gate = rng.uniform(0.0, 1.0)
            source_upper = minkowski @ (gate * source) @ minkowski
            projector = self.lorentzian_response_projector(wave)
            wavevectors[index] = wave
            responses[index] = np.einsum("mnab,ab->mn", projector, source_upper)
        return {"wavevectors": wavevectors, "responses": responses}

    def gate_response(
        self, invariant: np.ndarray | float, threshold: np.ndarray | float,
        power: np.ndarray | float,
    ) -> np.ndarray:
        ratio = np.power(np.asarray(invariant) / np.asarray(threshold), np.asarray(power))
        return ratio / (1.0 + ratio)

    def predict_gate_response(self, inputs: Record) -> Record:
        threshold, power = np.meshgrid(inputs["thresholds"], inputs["powers"])
        speed_of_light_km_s = self._speed_of_light_m_s / self._meters_per_kilometer
        invariants = (
            inputs["galaxy_invariant"],
            inputs["cmb_invariant"],
            (inputs["cluster_speed_km_s"] / speed_of_light_km_s) ** 2,
        )
        return {
            "responses": np.asarray(
                [self.gate_response(value, threshold, power) for value in invariants]
            ),
            "maximum_inactive_response": inputs["maximum_inactive_response"],
            "minimum_active_response": inputs["minimum_active_response"],
        }

    def predict_linear_stability(self, inputs: Record) -> Record:
        distribution = np.exp(-0.5 * inputs["momentum"] ** 2) / np.sqrt(2.0 * np.pi)
        gate = self.gate_response(
            inputs["invariants"], self.gate_threshold, self.gate_power
        )
        susceptibility = self.susceptibility(inputs["local_accelerations_m_s2"])
        tensor_poles = np.concatenate(
            [inputs["wave_numbers"], -inputs["wave_numbers"]]
        ).astype(complex)
        particle_speeds = np.abs(inputs["momentum"]) / np.sqrt(
            1.0 + inputs["momentum"] ** 2
        )
        characteristic_speeds = np.concatenate(
            [np.ones_like(inputs["wave_numbers"]), particle_speeds, np.zeros(1)]
        )
        roots = tuple(
            np.roots([1.0, 0.0, -(wave_number**2)])
            for wave_number in inputs["probe_wave_numbers"]
        )
        return {
            "momentum": inputs["momentum"],
            "distribution": distribution,
            "gate": gate,
            "susceptibility": susceptibility,
            "tensor_poles": tensor_poles,
            "characteristic_speeds": characteristic_speeds,
            "probe_wave_numbers": inputs["probe_wave_numbers"],
            "dispersion_roots": roots,
            "response_amplitude": self.response_h2,
        }

    def predict_particle_mesh_density(self, inputs: Record) -> np.ndarray:
        amplitude = self.response_to_baryon_ratio()
        positions = np.vstack([inputs["positions"].copy(), inputs["positions"].copy()])
        velocities = np.vstack(
            [inputs["velocities"].copy(), inputs["velocities"].copy()]
        )
        component_mass = np.asarray([1.0, amplitude])[:, None]
        return self._particle_mesh_engine.leapfrog(
            positions, velocities, component_mass, inputs
        )

    def predict_df4_response(self, state: Record) -> np.ndarray:
        """Apply the causal response to the shared DF4 nuisance draws."""
        external = self._infer_newtonian_acceleration(
            state["external_acceleration_m_s2"],
        )
        return self.response_rescaled_dispersion(
            state["baryonic_dispersion_km_s"],
            state["baryonic_acceleration_m_s2"],
            external,
        )

    def predict_jades_abundance(self, inputs: Record) -> Record:
        """Compute the JADES detection probability from the fixed matter endpoint."""
        h = self.reduced_hubble()
        omega_b = self.omega_b_h2 / h**2
        omega_m = (
            self.omega_b_h2
            + self.response_h2
            + self.neutrino_mass_ev / self._neutrino_mass_per_omega_h2_ev
        ) / h**2
        cosmology = FlatLambdaCDM(
            H0=self.hubble_km_s_mpc, Om0=omega_m, Ob0=omega_b,
            Tcmb0=self._cmb_temperature_k,
        )
        volume = float(
            (
                cosmology.comoving_volume(inputs["redshift_maximum"])
                - cosmology.comoving_volume(inputs["redshift_minimum"])
            ).value
            * (inputs["survey_area_arcmin2"] / self._arcmin2_per_degree2)
            / self._full_sky_square_degrees
        )
        parameters = self.build_camb_parameters(
            matter_redshifts=[inputs["target_redshift"]],
            matter_kmax=self._matter_power_kmax_mpc,
        )
        parameters.NonLinear = camb.model.NonLinear_none
        interpolator = camb.get_matter_power_interpolator(
            parameters, nonlinear=False, hubble_units=False, k_hunit=False,
            kmax=self._matter_power_kmax_mpc,
            zmax=inputs["target_redshift"] + self._matter_power_redshift_margin,
        )
        wave_number = np.geomspace(*self._sigma_wavenumber_grid_mpc)
        power = interpolator.P(inputs["target_redshift"], wave_number)
        masses = np.geomspace(*self._halo_mass_grid_msun)
        rho_critical = self._critical_density_msun_h2_mpc3 * h**2
        rho_matter = omega_m * rho_critical
        radii = (3.0 * masses / (4.0 * np.pi * rho_matter)) ** (1.0 / 3.0)
        sigma = np.empty_like(masses)
        for index, radius in enumerate(radii):
            argument = wave_number * radius
            window = np.ones_like(argument)
            selected = np.abs(argument) > self._top_hat_small_argument
            x = argument[selected]
            window[selected] = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
            variance = np.trapz(
                wave_number**3 * power * window**2, np.log(wave_number)
            ) / (2.0 * np.pi**2)
            sigma[index] = np.sqrt(max(variance, self._minimum_variance))
        nu = self._spherical_collapse_delta_c / sigma
        shape = self._sheth_tormen_a
        multiplicity = (
            self._sheth_tormen_normalization
            * np.sqrt(2.0 * shape / np.pi)
            * nu
            * np.exp(-0.5 * shape * nu**2)
            * (1.0 + (shape * nu**2) ** -self._sheth_tormen_p)
        )
        derivative = np.gradient(np.log(1.0 / sigma), np.log(masses))
        differential = rho_matter / masses * multiplicity * derivative
        cumulative = -cumulative_trapezoid(
            differential[::-1], np.log(masses[::-1]), initial=0.0
        )[::-1]
        baryon_fraction = omega_b / omega_m
        efficiencies = np.geomspace(*self._star_formation_efficiency_grid)
        minimum_halo_masses = inputs["stellar_mass_msun"] / (
            baryon_fraction * efficiencies
        )
        expected_scan = (
            np.interp(
                minimum_halo_masses, masses, cumulative,
                left=cumulative[0], right=0.0,
            )
            * volume
        )
        minimum_halo_mass = inputs["stellar_mass_msun"] / (
            baryon_fraction * inputs["star_formation_efficiency"]
        )
        expected = float(
            np.interp(inputs["star_formation_efficiency"], efficiencies, expected_scan)
        )
        probability = float(1.0 - np.exp(-expected))
        return {
            "value": probability,
            "details": {
                "source": inputs["source"],
                "target_redshift": inputs["target_redshift"],
                "secure_rows_z13p5_to_z14p5": inputs["secure_rows"],
                "geometric_volume_mpc3": volume,
                "minimum_halo_mass_msun": minimum_halo_mass,
                "expected_count": expected,
                "ontology_note": (
                    "The initialized pressureless response and reference CDM "
                    "have the same abundance prediction at this level."
                ),
            },
        }

    def _standardize_cluster_map(
        self, values: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        result = np.zeros_like(values, dtype=float)
        selected = values[mask]
        scale = max(np.std(selected), self._minimum_map_scale)
        result[mask] = (selected - np.mean(selected)) / scale
        return result

    def _transform_cluster_map(
        self, values: np.ndarray, mask: np.ndarray, percentiles: tuple[float, float]
    ) -> np.ndarray:
        background_percentile, scale_percentile = percentiles
        background = np.percentile(values[mask], background_percentile)
        positive = np.maximum(values - background, 0.0)
        scale = np.percentile(positive[mask], scale_percentile)
        transformed = np.arcsinh(positive / max(scale, self._minimum_map_scale))
        transformed[~mask] = 0.0
        return self._standardize_cluster_map(transformed, mask)

    def _cluster_features(
        self, light: np.ndarray, gas: np.ndarray, mask: np.ndarray,
        pixel_arcsec: float, redshift: float, smoothing_scales_kpc: tuple[float, ...],
    ) -> np.ndarray:
        components = []
        for source in (light, gas):
            standardized = self._standardize_cluster_map(source, mask)
            rows = [standardized[mask]]
            kpc_per_arcmin = Planck18.kpc_proper_per_arcmin(redshift).value
            for scale in smoothing_scales_kpc:
                sigma = (
                    self._arcsec_per_arcmin * scale / kpc_per_arcmin
                ) / pixel_arcsec
                rows.append(
                    self._standardize_cluster_map(
                        gaussian_filter(standardized, sigma=sigma), mask
                    )[mask]
                )
            components.extend(rows)
        base = np.column_stack(components)
        offset = len(smoothing_scales_kpc) + 1
        baryon_local = (base[:, 0] + base[:, offset]) / np.sqrt(2.0)
        return np.column_stack([base, baryon_local])

    def _prepare_hff_cluster(
        self, item: Record, smoothing_scales_kpc: tuple[float, ...]
    ) -> Record:
        mask = np.isfinite(item["merten"])
        for footprint in item["footprints"]:
            mask &= footprint > self._hff_footprint_threshold
        percentiles = self._hff_map_percentiles
        smoothing = self._map_smoothing_arcsec / item["pixel_arcsec"]
        light = gaussian_filter(np.nan_to_num(item["light"], nan=0.0), sigma=smoothing)
        gas = gaussian_filter(np.nan_to_num(item["gas"], nan=0.0), sigma=smoothing)
        targets = {
            "Sharon": self._transform_cluster_map(item["sharon"], mask, percentiles)[mask],
            "Merten": self._transform_cluster_map(item["merten"], mask, percentiles)[mask],
        }
        return {
            "features": self._cluster_features(
                self._transform_cluster_map(light, mask, percentiles),
                self._transform_cluster_map(gas, mask, percentiles),
                mask, item["pixel_arcsec"], item["redshift"], smoothing_scales_kpc,
            ),
            "targets": targets,
        }

    def _prepare_abell520_cluster(
        self, item: Record, smoothing_scales_kpc: tuple[float, ...]
    ) -> Record:
        catalog = item["member_catalogue"]
        minimum_redshift, maximum_redshift = A520_MEMBER_REDSHIFT_RANGE
        selected = (
            (catalog[:, A520_REDSHIFT_COLUMN] >= minimum_redshift)
            & (catalog[:, A520_REDSHIFT_COLUMN] <= maximum_redshift)
            & (catalog[:, A520_REDSHIFT_ODDS_COLUMN] > A520_MINIMUM_REDSHIFT_ODDS)
            & (catalog[:, A520_STELLARITY_COLUMN] < A520_MAXIMUM_STELLARITY)
            & (catalog[:, A520_MAGNITUDE_COLUMN] < A520_MAXIMUM_MAGNITUDE)
            & (catalog[:, A520_FLUX_NJY_COLUMN] > 0)
        )
        members = catalog[selected]
        light = np.zeros_like(item["right_ascension"])
        x = np.interp(
            members[:, A520_RA_COLUMN],
            item["right_ascension"][0],
            np.arange(item["right_ascension"].shape[1]),
        )
        y = np.interp(
            members[:, A520_DEC_COLUMN],
            item["declination"][:, 0],
            np.arange(item["declination"].shape[0]),
        )
        flux = members[:, A520_FLUX_NJY_COLUMN]
        weights = np.minimum(
            flux, np.percentile(flux, self._abell520_weight_percentile_cap)
        )
        for px, py, weight in zip(x, y, weights):
            if 0 <= px < light.shape[1] and 0 <= py < light.shape[0]:
                light[int(round(py)), int(round(px))] += weight
        smoothing = self._map_smoothing_arcsec / item["pixel_arcsec"]
        light = gaussian_filter(light, sigma=smoothing)
        gas = gaussian_filter(item["gas"], sigma=smoothing)
        support = light > np.percentile(
            light[light > 0], self._abell520_light_support_percentile
        )
        support = (
            gaussian_filter(
                support.astype(float), sigma=self._abell520_support_smoothing_pixels
            )
            > self._abell520_minimum_support_weight
        )
        mask = (
            item["gas_footprint"] & support & np.isfinite(item["clowe_lensing"])
        )
        percentiles = self._abell520_map_percentiles
        normalized_light = self._transform_cluster_map(light, mask, percentiles)
        normalized_gas = self._transform_cluster_map(gas, mask, percentiles)
        return {
            "features": self._cluster_features(
                normalized_light, normalized_gas, mask,
                item["pixel_arcsec"], item["redshift"], smoothing_scales_kpc,
            ),
            "targets": {
                "Jee 2014": self._transform_cluster_map(
                    item["jee_lensing"], mask, percentiles
                )[mask],
                "Clowe 2012": self._transform_cluster_map(
                    item["clowe_lensing"], mask, percentiles
                )[mask],
            },
        }

    def _fit_cluster_proxy(
        self, prepared: dict[str, Record], systems: list[str], selected: np.ndarray
    ) -> np.ndarray:
        feature_rows = []
        target_rows = []
        weight_rows = []
        for system in systems:
            item = prepared[system]
            features = item["features"]
            targets = item["targets"]
            count = features.shape[0]
            weight = 1.0 / (len(targets) * count)
            for target in targets.values():
                feature_rows.append(features[:, selected])
                target_rows.append(target)
                weight_rows.append(np.full(count, weight))
        features = np.vstack(feature_rows)
        target = np.concatenate(target_rows)
        sqrt_weight = np.sqrt(np.concatenate(weight_rows))
        return nnls(features * sqrt_weight[:, None], target * sqrt_weight)[0]

    def _project_cluster_targets(
        self, prepared: dict[str, Record], system: str, selected: np.ndarray,
        coefficient: np.ndarray,
    ) -> tuple[Record, ...]:
        return tuple(
            {
                "reconstruction": reconstruction,
                "observed": target,
                "predicted": prepared[system]["features"][:, selected] @ coefficient,
            }
            for reconstruction, target in prepared[system]["targets"].items()
        )

    def predict_cluster_proxies(self, inputs: Record) -> Record:
        """Fit nested baryon-map proxies and predict each held-out cluster."""
        scales = self.cluster_smoothing_scales_kpc
        prepared = {
            item["name"]: self._prepare_hff_cluster(item, scales)
            for item in inputs["hff_systems"]
        }
        abell520 = inputs["abell520"]
        prepared[abell520["name"]] = self._prepare_abell520_cluster(abell520, scales)
        systems = tuple(prepared)
        local_feature = len(self._cluster_map_kinds) * (len(scales) + 1)
        selections = tuple(
            (f"factorized_sigma_{int(scale)}_kpc", np.asarray([local_feature, index]))
            for index, scale in enumerate(scales, start=1)
        )
        held_predictions = []
        for held in systems:
            outer = [system for system in systems if system != held]
            nested_by_candidate: dict[str, tuple[Record, ...]] = {}
            for candidate_name, selected in selections:
                nested = []
                for inner_held in outer:
                    training = [system for system in outer if system != inner_held]
                    coefficient = self._fit_cluster_proxy(prepared, training, selected)
                    nested.extend(
                        self._project_cluster_targets(
                            prepared, inner_held, selected, coefficient
                        )
                    )
                nested_by_candidate[candidate_name] = tuple(nested)
            candidates = []
            for candidate_name, selected in selections:
                coefficient = self._fit_cluster_proxy(prepared, outer, selected)
                held_rows = self._project_cluster_targets(
                    prepared, held, selected, coefficient
                )
                candidates.append({
                    "name": candidate_name,
                    "nested": nested_by_candidate[candidate_name],
                    "held": held_rows,
                })
            held_predictions.append({
                "held_system": held,
                "candidates": tuple(candidates),
            })
        return {"held_systems": tuple(held_predictions), "source": inputs["source"]}
