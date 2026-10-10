from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from causal_baryon_gravity.benchmarks import headline_summary, write_summary


class HeadlineTests(unittest.TestCase):
    def test_manuscript_headlines(self) -> None:
        summary = headline_summary()
        self.assertEqual(summary["sparc"]["radii"], 3389)
        self.assertEqual(summary["sparc"]["galaxies"], 175)
        self.assertAlmostEqual(
            summary["sparc"]["rmse_km_s"], 22.811405517468, places=12
        )
        self.assertAlmostEqual(
            summary["sparc"]["canonical_only_rmse_km_s"],
            22.807528591682,
            places=12,
        )
        self.assertAlmostEqual(
            summary["bullet"]["response_to_baryon_normalization"],
            6.361930294906166,
            places=15,
        )
        self.assertAlmostEqual(
            summary["bullet"]["centroid_kpc"], 86.639275, places=6
        )
        self.assertAlmostEqual(
            summary["bullet"]["aperture_deviance"], 2.657774, places=6
        )
        self.assertAlmostEqual(summary["cluster_maps"]["mean"], 0.739675, places=6)
        self.assertAlmostEqual(
            summary["cluster_maps"]["hff_minimum"], 0.674837, places=6
        )
        self.assertAlmostEqual(
            summary["cluster_maps"]["a520_minimum"], 0.481154, places=6
        )
        self.assertAlmostEqual(
            summary["cosmology_transfer_estimates"]["response_fraction"],
            0.8390189009861309,
            places=15,
        )
        self.assertIn(
            "transfer-adapter estimates",
            summary["cosmology_transfer_estimates"]["interpretation"],
        )
        self.assertAlmostEqual(
            summary["limitations"]["df2_fixed_law_dispersion_km_s"],
            21.113989,
            places=6,
        )
        self.assertAlmostEqual(
            summary["limitations"]["df4_fixed_law_dispersion_km_s"],
            19.757741,
            places=6,
        )

    def test_summary_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_summary(output)
            for name in (
                "summary.json",
                "summary.csv",
                "headline_summary.png",
                "acceleration_profile.csv",
                "acceleration_profile.png",
            ):
                self.assertGreater((output / name).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
