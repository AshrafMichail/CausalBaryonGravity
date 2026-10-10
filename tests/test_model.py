from __future__ import annotations

import unittest

import numpy as np

from causal_baryon_gravity import (
    A_MEM,
    A0_M_S2,
    SIGMA_H_KPC_COMOVING,
    CausalBaryonGravity,
)


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = CausalBaryonGravity()

    def test_reproducible_constants(self) -> None:
        self.assertEqual(self.model.a0_m_s2, A0_M_S2)
        self.assertEqual(A0_M_S2, 1.2e-10)
        self.assertEqual(self.model.memory_amplitude, A_MEM)
        self.assertEqual(A_MEM, 5.361930294906166)
        self.assertEqual(self.model.sigma_h_kpc_comoving, SIGMA_H_KPC_COMOVING)
        self.assertEqual(SIGMA_H_KPC_COMOVING, 150.0)

    def test_formula_matches_literal_expression(self) -> None:
        baryonic = np.array([1.0e-13, 1.0e-11, 1.0e-9])
        radius = np.array([1.0, 150.0, 900.0])
        memory = (
            np.vectorize(__import__("math").erf)(
                radius / (np.sqrt(2.0) * SIGMA_H_KPC_COMOVING)
            )
            - np.sqrt(2.0 / np.pi)
            * (radius / SIGMA_H_KPC_COMOVING)
            * np.exp(-radius**2 / (2.0 * SIGMA_H_KPC_COMOVING**2))
        )
        expected = baryonic * (
            0.5
            + np.sqrt(0.25 + A0_M_S2 / baryonic)
            + A_MEM * memory
        )
        np.testing.assert_allclose(
            self.model.acceleration(baryonic, radius),
            expected,
            rtol=2.0e-15,
            atol=0.0,
        )

    def test_limiting_behavior(self) -> None:
        self.assertEqual(float(self.model.acceleration(0.0, 0.0)), 0.0)
        self.assertEqual(float(self.model.memory_profile(0.0)), 0.0)
        self.assertAlmostEqual(
            float(self.model.memory_profile(1.0e7)), 1.0, places=15
        )
        high_acceleration = 1.0
        ratio = float(
            self.model.acceleration(high_acceleration, 1.0e7)
            / high_acceleration
        )
        self.assertAlmostEqual(
            ratio, self.model.response_to_baryon_normalization, places=9
        )

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.model.acceleration(-1.0, 1.0)
        with self.assertRaises(ValueError):
            self.model.acceleration(1.0, -1.0)
        with self.assertRaises(ValueError):
            CausalBaryonGravity(sigma_h_kpc_comoving=0.0)
        with self.assertRaises(ValueError):
            self.model.circular_speed_km_s(1.0e-11, 1.0, -1.0)

    def test_physical_radius_is_explicit_for_speed(self) -> None:
        speed = self.model.circular_speed_km_s(1.0e-11, 150.0, 100.0)
        acceleration = self.model.acceleration(1.0e-11, 150.0)
        expected = np.sqrt(acceleration * 100.0 * 3.085677581491367e19) / 1000.0
        self.assertEqual(float(speed), float(expected))


if __name__ == "__main__":
    unittest.main()
