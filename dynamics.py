"""Shared Newtonian transformations used by independent model classes."""

from __future__ import annotations

import numpy as np


class NewtonianDynamics:
    """Canonical unit conversions and compact-system Newtonian mechanics."""

    _g_kpc_kms2_msun = 4.30091e-6
    _kms2_per_kpc_to_ms2 = 3.240779289e-14

    def acceleration_from_speed(
        self, speed_km_s: np.ndarray | float, radius_kpc: np.ndarray | float
    ) -> np.ndarray:
        """Convert circular speed at a radius into centripetal acceleration."""
        return (np.square(np.asarray(speed_km_s)) / np.asarray(radius_kpc)
                * self._kms2_per_kpc_to_ms2)

    def speed_from_acceleration(
        self, acceleration_m_s2: np.ndarray | float, radius_kpc: np.ndarray | float
    ) -> np.ndarray:
        """Convert radial acceleration into the corresponding circular speed."""
        return np.sqrt(np.asarray(acceleration_m_s2) * np.asarray(radius_kpc)
                       / self._kms2_per_kpc_to_ms2)

    def point_mass_acceleration(
        self, enclosed_mass_msun: np.ndarray | float, radius_kpc: np.ndarray | float
    ) -> np.ndarray:
        """Evaluate Newtonian acceleration from enclosed mass and radius."""
        return (self._g_kpc_kms2_msun * np.asarray(enclosed_mass_msun)
                / np.square(np.asarray(radius_kpc))
                * self._kms2_per_kpc_to_ms2)

    def compact_radius(self, effective_radius_kpc: np.ndarray | float) -> np.ndarray:
        """Map projected effective radius to the adopted dynamical radius."""
        return 4.0 * np.asarray(effective_radius_kpc) / 3.0

    def compact_enclosed_mass(self, stellar_mass_msun: np.ndarray | float) -> np.ndarray:
        """Use the half-light radius convention to select enclosed stellar mass."""
        return 0.5 * np.asarray(stellar_mass_msun)

    def baryonic_dispersion(
        self, enclosed_mass_msun: np.ndarray | float, radius_kpc: np.ndarray | float
    ) -> np.ndarray:
        """Estimate pressure-supported dispersion from baryons alone."""
        return np.sqrt(self._g_kpc_kms2_msun * np.asarray(enclosed_mass_msun)
                       / (3.0 * np.asarray(radius_kpc)))
