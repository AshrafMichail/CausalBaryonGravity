"""Focused architecture, source-pinning, reporting, and parity tests."""

from __future__ import annotations

import ast
import inspect
import json
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np

from . import clusters, empirical, identities, validators
from .benchmarks import BaryonsOnlyModel, MONDModel, NFWModel
from .downloader import selected_sources, verify_present_inputs
from .model import CausalBaryonModel
from .orchestrator import PredictionOrchestrator
from .particle_mesh import ReferenceParticleMeshModel
from .records import BULLET_APERTURE_COLUMNS
from .registry import (
    CASES, LEDGER_CASES, REFERENCES, VALIDATION_CASES, execute_pipeline,
)
from .released import ReleasedEndpointModel
from .result import Result
from .run_validation import main
from .sources import SOURCES
from .validators import ValidationMetadata

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
EXISTING_RAW = HERE.parent / "data" / "raw"
ADAPTER_FILES = ("empirical.py", "identities.py", "clusters.py")
MODEL_MODULES = {
    "model", "benchmarks", "dynamics", "orchestrator", "particle_mesh", "released"
}
PYTHON_LINE_CEILING = 3600
# Magnitudes that may stay inline: the additive and multiplicative identities,
# one half, and the small integers used by standard formulas, tensor ranks, and
# slice bounds. Everything else must come from a named constant or field.
UNIVERSAL_MAGNITUDES = frozenset({0, 0.5, 1, 2, 3, 4})
DECIMAL_BASE = 10
LITERAL_CONTROL_SOURCE = """
LIMIT_SECONDS = 3600.0


class Model:
    scale: float = 1.2e-10

    def method(self, value):
        return value * self.scale / LIMIT_SECONDS - 0.5 + 10 ** 2 + 26.0
"""


def imported_modules(path: Path) -> set[str]:
    imported = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    return imported


class FunctionLiteralAudit(ast.NodeVisitor):
    """Collect numeric literals written directly inside function bodies.

    Module and class assignments, including dataclass field defaults, are the
    intended home of every configuration value, so only code inside functions
    is audited. Universal magnitudes stay inline, as does the decimal base of
    an explicit ``10 ** x``. There is no per-file or per-line exemption.
    """

    def __init__(self) -> None:
        self.depth = 0
        self.offenders: list[tuple[int, complex]] = []

    def visit_FunctionDef(self, node: ast.AST) -> None:
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef

    def visit_BinOp(self, node: ast.BinOp) -> None:
        base = node.left
        decimal_power = isinstance(base, ast.Constant) and base.value == DECIMAL_BASE
        if isinstance(node.op, ast.Pow) and decimal_power:
            self.visit(node.right)
            return
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        value = node.value
        if not self.depth or isinstance(value, bool):
            return
        if isinstance(value, (int, float, complex)):
            if abs(value) not in UNIVERSAL_MAGNITUDES:
                self.offenders.append((node.lineno, value))


def unnamed_literals(source: str) -> list[tuple[int, complex]]:
    audit = FunctionLiteralAudit()
    audit.visit(ast.parse(source))
    return audit.offenders


class ReviewValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        verify_present_inputs(EXISTING_RAW, list(VALIDATION_CASES))
        application = PredictionOrchestrator()
        cls.results = {
            name: spec.bind(application).run(EXISTING_RAW)
            for name, spec in CASES.items()
        }

    def test_ledger_pipeline_and_sentinel_flow(self) -> None:
        self.assertEqual((len(LEDGER_CASES), len(VALIDATION_CASES)), (22, 23))
        self.assertEqual(tuple(CASES), VALIDATION_CASES)
        self.assertEqual(VALIDATION_CASES[-1], "df4")
        application = PredictionOrchestrator()
        for name, spec in CASES.items():
            with self.subTest(case=name):
                self.assertIn(
                    spec.adapter.__module__.rsplit(".", 1)[-1],
                    {"empirical", "identities", "clusters"},
                )
                self.assertTrue(spec.model_method.__qualname__.startswith(
                    "PredictionOrchestrator."
                ))
                self.assertIs(spec.bind(application).model_method.__self__, application)
                self.assertEqual(spec.validator.__module__, validators.__name__)

        adapted, predicted = object(), object()
        observed: list[object] = []
        metadata = ValidationMetadata("sentinel", "test", "flow", "value", 1.0)

        def model(value: object) -> object:
            self.assertIs(value, adapted)
            return predicted

        def validate(value: object, received: ValidationMetadata) -> Result:
            observed.append(value)
            return Result(
                received.name, received.evidence, received.method,
                received.metric, 1.0, received.expected_value,
            )

        result = execute_pipeline(lambda _: adapted, model, validate, metadata, HERE)
        self.assertIs(observed[0], predicted)
        self.assertEqual(result.value, 1.0)

    def test_adapter_validator_and_repository_boundaries(self) -> None:
        for filename in ADAPTER_FILES:
            path = HERE / filename
            source = path.read_text()
            with self.subTest(adapter=filename):
                self.assertTrue(MODEL_MODULES.isdisjoint(imported_modules(path)))
                for term in (
                    "CausalBaryonModel", "PredictionOrchestrator",
                    "a0_m_s2", "omega_b_h2", "response_h2",
                ):
                    self.assertNotIn(term, source)

        validation_source = (HERE / "validators.py").read_text()
        self.assertTrue(
            MODEL_MODULES.isdisjoint(imported_modules(HERE / "validators.py"))
        )
        for term in (
            "CausalBaryonModel", "PredictionOrchestrator", "camb", "astropy",
            "nnls", "a0_m_s2", "omega_b_h2", "response_h2",
        ):
            self.assertNotIn(term, validation_source)

        first = empirical.galaxy_dynamics(EXISTING_RAW)
        second = empirical.galaxy_dynamics(EXISTING_RAW)
        np.testing.assert_array_equal(first["radius_kpc"], second["radius_kpc"])
        np.testing.assert_array_equal(
            first["external_fields"]["galaxies"],
            second["external_fields"]["galaxies"],
        )
        self.assertTrue(np.any(
            first["external_fields"]["fitted_log10_acceleration"] >= -10.6
        ))

        forbidden = (
            "darkmatter/paper", "darkmatter/src", "darkmatter/outputs",
            "from src", "kernel_validation", "metrics.json", "generated_result",
            "lru_cache", "_MEMO",
        )
        for path in HERE.glob("*.py"):
            if path.name == Path(__file__).name:
                continue
            with self.subTest(isolation=path.name):
                source = path.read_text()
                self.assertFalse(any(term in source for term in forbidden))
                self.assertNotIn("importlib", source)
                self.assertNotIn("__import__", source)

    def test_explicit_models_and_formula_ownership(self) -> None:
        application = PredictionOrchestrator()
        expected_types = {
            "causal_model": CausalBaryonModel,
            "mond_model": MONDModel,
            "nfw_model": NFWModel,
            "baryons_only_model": BaryonsOnlyModel,
            "reference_particle_mesh_model": ReferenceParticleMeshModel,
            "released_endpoint_model": ReleasedEndpointModel,
        }
        for field, expected in expected_types.items():
            self.assertIs(type(getattr(application, field)), expected)
        self.assertIsNot(application.mond_model, application.nfw_model)

        causal = inspect.getsource(CausalBaryonModel)
        benchmark = (HERE / "benchmarks.py").read_text()
        for term in (
            "mond", "nfw", "benchmark", "canonical_mass",
            "flexible_mass", "concentration_log10",
        ):
            self.assertNotIn(term, causal.lower())
        self.assertNotIn("predict_sdss", causal)
        self.assertEqual(causal.count("self.response_h2 / self.omega_b_h2"), 1)
        self.assertEqual(causal.count("self._algebraic_external_response_nu("), 1)
        self.assertEqual(causal.count("self._infer_newtonian_acceleration("), 1)
        self.assertEqual(
            benchmark.count("offset + np.sqrt(offset**2 + 1.0 / ratio)"), 1
        )
        self.assertEqual(benchmark.count("np.log1p(scaled_radius)"), 1)
        self.assertEqual(MONDModel().acceleration_scale_m_s2, 1.2e-10)
        baryons = BaryonsOnlyModel()
        self.assertEqual(
            (baryons.disk_mass_to_light, baryons.bulge_mass_to_light), (0.5, 0.7)
        )

        model_tree = ast.parse((HERE / "model.py").read_text())
        self.assertEqual(
            [node.name for node in model_tree.body if isinstance(node, ast.ClassDef)],
            ["CausalBaryonModel"],
        )
        self.assertFalse(
            any(isinstance(node, ast.FunctionDef) for node in model_tree.body)
        )
        orchestrator_tree = ast.parse((HERE / "orchestrator.py").read_text())
        arithmetic = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.MatMult)
        self.assertFalse(any(
            isinstance(node, ast.BinOp) and isinstance(node.op, arithmetic)
            for node in ast.walk(orchestrator_tree)
        ))
        self.assertNotIn("numpy", imported_modules(HERE / "orchestrator.py"))
        for term in ("response_h2", "omega_b_h2", "1.45947e-10"):
            self.assertNotIn(term, benchmark)
        for path in HERE.glob("*.py"):
            imports = imported_modules(path)
            if "camb" in imports or "astropy.cosmology" in imports:
                self.assertEqual(path.name, "model.py")

        baryonic = np.array([1e-12, 1e-10, 1e-8])
        central = application.causal_model.galaxy_acceleration(baryonic)
        self.assertTrue(np.all(central >= baryonic))
        self.assertNotEqual(
            float(application.mond_model.acceleration(1e-11)),
            float(application.causal_model.galaxy_acceleration(1e-11)),
        )
        with self.assertRaises(ValueError):
            application.causal_model.galaxy_acceleration(-1.0)

        external = np.array([0.0, 1e-13, 1e-11])
        isolated = (
            np.sqrt(0.25 + application.causal_model.a0_m_s2 / baryonic)
            - 0.5
        )
        screen = 1.0 / (
            1.0
            + (
                external
                / application.causal_model.external_acceleration_m_s2
            )
            ** application.causal_model.gate_power
        )
        expected = baryonic * (
            1.0 + application.causal_model.alpha * isolated * screen
        )
        np.testing.assert_array_equal(
            application.causal_model.galaxy_acceleration(
                baryonic,
                external,
            ),
            expected,
        )
        dynamic = application.causal_model.galaxy_acceleration(baryonic)
        np.testing.assert_allclose(
            application.causal_model._infer_newtonian_acceleration(dynamic),
            baryonic,
            rtol=1e-14,
            atol=0.0,
        )

    def test_configuration_values_are_named_constants(self) -> None:
        """Reject numeric literals that reappear inside production functions."""
        self.assertEqual(unnamed_literals(LITERAL_CONTROL_SOURCE), [(9, 26.0)])
        offenders = {
            path.name: unnamed_literals(path.read_text())
            for path in sorted(HERE.glob("*.py"))
            if path.name != Path(__file__).name
        }
        self.assertEqual({name: rows for name, rows in offenders.items() if rows}, {})

        aperture = identities.bullet_aperture_peaks(EXISTING_RAW)
        self.assertEqual(aperture["columns"], BULLET_APERTURE_COLUMNS)
        self.assertEqual(
            aperture["measurements"].shape,
            (len(identities.BULLET_SYSTEM_NAMES),
             len(identities.BULLET_POSITION_NAMES), len(BULLET_APERTURE_COLUMNS)),
        )
        self.assertEqual(empirical.JADES_SECURE_REDSHIFT_RANGE, (13.5, 14.5))
        self.assertEqual(clusters.ARCSEC_PER_DEGREE, 3600.0)
        self.assertEqual(
            CausalBaryonModel().parameters()["cluster_smoothing_scales_kpc"],
            [50.0, 100.0, 150.0, 200.0],
        )

    def test_downloader_sources_are_explicitly_pinned(self) -> None:
        self.assertEqual(len(SOURCES), 55)
        self.assertEqual(len({source.path for source in SOURCES}), 55)
        self.assertEqual(selected_sources(list(VALIDATION_CASES)), SOURCES)
        for source in SOURCES:
            with self.subTest(path=source.path):
                self.assertTrue(source.url.startswith("https://"))
                self.assertRegex(source.sha256, r"^[0-9a-f]{64}$")
                self.assertTrue(source.cases)
                self.assertNotIn("runtime_output", source.path)
        a520 = next(source for source in SOURCES if source.path.endswith(
            "lensing/kappa_j14_lambda3.0.fits"
        ))
        self.assertIn("1dd39bcd33917f507c668d22a85dc848931acc95", a520.url)

    def test_all_23_scalar_results_and_key_details(self) -> None:
        self.assertEqual(tuple(self.results), VALIDATION_CASES)
        for name, result in self.results.items():
            expected, tolerance = REFERENCES[name]
            with self.subTest(case=name):
                self.assertEqual(result.name, name)
                self.assertEqual(result.expected_value, expected)
                self.assertEqual(result.value, expected)

        galaxy = self.results["galaxy_dynamics"].details
        self.assertAlmostEqual(
            galaxy["fixed_simple_mond_rmse_km_s"], 22.807528591681837, places=12
        )
        self.assertEqual(galaxy["galaxies_with_external_field"], 90)
        bullet = self.results["bullet_aperture_peaks"].details
        self.assertAlmostEqual(
            bullet["subcluster_predicted_plasma_kappa"],
            0.022394368937620222, places=14,
        )
        self.assertAlmostEqual(
            bullet["subcluster_bcg_to_plasma_ratio"], 8.88788332346543, places=12
        )
        df4 = self.results["df4"].details
        self.assertAlmostEqual(
            df4["canonical_nfw_delta_deviance"], 11.7475748031746, places=12
        )
        self.assertAlmostEqual(
            df4["flexible_nfw_delta_deviance"], 2.611966539773462, places=12
        )

    def test_missing_data_report_and_runtime_ignores(self) -> None:
        root = HERE / "runtime_data" / "test_missing"
        output = HERE / "runtime_output" / "test_missing"
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(output, ignore_errors=True)
        self.addCleanup(shutil.rmtree, root, True)
        self.addCleanup(shutil.rmtree, output, True)
        status = main([
            "--data-dir", str(root), "--output-dir", str(output),
            "--cases", "galaxy_dynamics",
        ])
        payload = json.loads((output / "summary.json").read_text())
        self.assertEqual(status, 2)
        self.assertFalse(payload["coverage"]["complete"])
        self.assertIn("MassModels_Lelli2016c.mrt", payload["skipped"]["galaxy_dynamics"])
        ignores = (HERE / ".gitignore").read_text()
        self.assertIn("runtime_data/", ignores)
        self.assertIn("runtime_output/", ignores)

    def test_flat_package_and_python_loc_ceiling(self) -> None:
        tracked = subprocess.check_output(
            ["git", "-C", str(REPOSITORY), "ls-files", "darkmatter/review_validation"],
            text=True,
        ).splitlines()
        prefix = "darkmatter/review_validation/"
        self.assertTrue(tracked)
        for path in tracked:
            relative = path[len(prefix):]
            self.assertNotIn("/", relative)
            self.assertNotIn("runtime_", relative)
        nested_python = [
            path for path in HERE.rglob("*.py")
            if path.parent != HERE and "__pycache__" not in path.parts
        ]
        self.assertEqual(nested_python, [])
        counts = {
            path.name: len(path.read_text().splitlines()) for path in HERE.glob("*.py")
        }
        self.assertLessEqual(sum(counts.values()), PYTHON_LINE_CEILING, counts)


if __name__ == "__main__":
    unittest.main()
