"""Closed-form spherical weak-field Causal Baryon Gravity law."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erf

A0_M_S2 = 1.2e-10
A_MEM = 5.361930294906166
SIGMA_H_KPC_COMOVING = 150.0
KPC_M = 3.085677581491367e19


@dataclass(frozen=True)
class CausalBaryonGravity:
    """Evaluate the manuscript's fixed spherical weak-field law."""

    a0_m_s2: float = A0_M_S2
    memory_amplitude: float = A_MEM
    sigma_h_kpc_comoving: float = SIGMA_H_KPC_COMOVING

    def __post_init__(self) -> None:
        if self.a0_m_s2 <= 0.0:
            raise ValueError("a0_m_s2 must be positive")
        if self.memory_amplitude < 0.0:
            raise ValueError("memory_amplitude must be nonnegative")
        if self.sigma_h_kpc_comoving <= 0.0:
            raise ValueError("sigma_h_kpc_comoving must be positive")

    def memory_profile(
        self, radius_kpc_comoving: np.ndarray | float
    ) -> np.ndarray:
        """Return the normalized spherical Gaussian enclosed-memory profile."""
        radius = np.asarray(radius_kpc_comoving, dtype=float)
        if np.any(~np.isfinite(radius)) or np.any(radius < 0.0):
            raise ValueError("radius_kpc_comoving must be finite and nonnegative")
        scaled = radius / self.sigma_h_kpc_comoving
        return (
            erf(scaled / np.sqrt(2.0))
            - np.sqrt(2.0 / np.pi) * scaled * np.exp(-0.5 * scaled**2)
        )

    def acceleration(
        self,
        baryonic_acceleration_m_s2: np.ndarray | float,
        radius_kpc_comoving: np.ndarray | float,
    ) -> np.ndarray:
        """Return g_CBG in m/s^2.

        The algebraically equivalent product form used here remains finite at
        g_b=0 and has the correct zero-acceleration limit.
        """
        baryonic, radius = np.broadcast_arrays(
            np.asarray(baryonic_acceleration_m_s2, dtype=float),
            np.asarray(radius_kpc_comoving, dtype=float),
        )
        if np.any(~np.isfinite(baryonic)) or np.any(baryonic < 0.0):
            raise ValueError(
                "baryonic_acceleration_m_s2 must be finite and nonnegative"
            )
        memory = self.memory_profile(radius)
        canonical = (
            0.5 * baryonic
            + np.sqrt(0.25 * baryonic**2 + self.a0_m_s2 * baryonic)
        )
        return canonical + baryonic * self.memory_amplitude * memory

    def circular_speed_km_s(
        self,
        baryonic_acceleration_m_s2: np.ndarray | float,
        radius_kpc_comoving: np.ndarray | float,
        radius_kpc_physical: np.ndarray | float,
    ) -> np.ndarray:
        """Convert acceleration to speed using an explicitly physical radius."""
        physical = np.asarray(radius_kpc_physical, dtype=float)
        if np.any(~np.isfinite(physical)) or np.any(physical < 0.0):
            raise ValueError("radius_kpc_physical must be finite and nonnegative")
        acceleration = self.acceleration(
            baryonic_acceleration_m_s2, radius_kpc_comoving
        )
        return np.sqrt(acceleration * physical * KPC_M) / 1000.0

    @property
    def response_to_baryon_normalization(self) -> float:
        """Asymptotic multiplicative response including the baryonic unit term."""
        return 1.0 + self.memory_amplitude
