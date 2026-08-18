# Causal Baryon review validation

This directory is a flat, self-contained review package. It contains the fixed
22-row paper ledger and the DF4 extension. It never reads prior generated
summaries: every selected case is recalculated in the current process and every
summary and PNG is newly written to `runtime_output/`.

## Compact review path

For a compact human review, follow:

1. `registry.py`: the single declarative source for names, references,
   tolerances, evidence, methods, adapters, applications, and validators.
2. `empirical.py`, `identities.py`, and `clusters.py`: parameter-free adapters.
3. `orchestrator.py`: short applications of one explicit instance of each
   model.
4. `model.py`, then `benchmarks.py`, `dynamics.py`, `particle_mesh.py`, and
   `released.py`: causal equations and independent comparison calculations.
5. `validators.py`: physics-free scalar, vector, covariance, deviance,
   correlation, and model-specific statistical checks.
6. `run_validation.py`: checksum verification, execution, reports, and plots.

Adapter inputs and predictions use the one generic typed mapping in
`records.py`, which also holds the shared published-array schemas; schema-only
dataclasses were removed.

## Named configuration values

Every domain or configuration value is defined once, with a descriptive name,
next to the module that owns it: model and comparison parameters are dataclass
fields or private class constants (`model.py`, `benchmarks.py`, `dynamics.py`),
released-product page ranges, header rows, band columns, catalogue columns and
sample cuts are module constants (`empirical.py`, `clusters.py`, `model.py`),
published measurements and synthetic probe settings are module constants
(`identities.py`), and statistical guards, download and plotting options live
with their own module (`validators.py`, `downloader.py`, `run_validation.py`).
Only universal mathematical magnitudes (`0`, `±1`, `2`, `3`, `4`, `1/2`, and
the decimal base of `10 ** x`) remain inline; the focused tests fail on any
other numeric literal written inside a function body.

## Architecture

Every registry entry follows one explicit path:

`input adapter -> bound PredictionOrchestrator method -> statistical validator -> Result`

`empirical.py`, `identities.py`, and `clusters.py` only read released products
or package published/synthetic probe inputs. `run_validation.py` constructs one
`PredictionOrchestrator` and binds every `CaseSpec` in `registry.py` to that
same instance. The orchestrator owns exactly one `CausalBaryonModel`, plus
separate `MONDModel`, `NFWModel`,
`BaryonsOnlyModel`, `ReferenceParticleMeshModel`, and
`ReleasedEndpointModel` instances. It only delegates and packages typed
outputs; it contains no physical equations.

The model layer remains flat:

- `model.py` owns only causal-response equations and forward calculations.
- `benchmarks.py` owns the explicit MOND and NFW equations and priors, plus
  the baryons-only galaxy and DF4 baselines.
- `released.py` replays the released SDSS endpoint independently of the
  causal model.
- `dynamics.py` owns the one shared Newtonian unit/compact-system
  implementation.
- `particle_mesh.py` owns one CIC deposition, gather, force, and leapfrog
  implementation used by both particle representations.
- `records.py` defines the neutral mapping exchanged between all three layers.
- `validators.py` only computes residual statistics, likelihood aggregates,
  correlations, flags, and `Result` objects.

The package public API is:

- orchestration: `PredictionOrchestrator`;
- causal model: `CausalBaryonModel`, including
  `response_to_baryon_ratio`, `susceptibility`, `galaxy_acceleration`,
  `compact_dispersion`, `cosmology_state`, `reduced_hubble`,
  `build_camb_parameters`, and its causal forward predictions;
- comparisons: `MONDModel`, `NFWModel`, `BaryonsOnlyModel`,
  `ReferenceParticleMeshModel`, and `ReleasedEndpointModel`;
- cosmology predictions:
  `predict_planck_spectra`, `predict_planck_lensing`,
  and `predict_jades_abundance`;
- clusters and compact systems:
  `predict_bullet_centroid`,
  `predict_bullet_apertures`, `predict_cluster_proxies`,
  and composed `predict_df4`;
- identities and numerical probes: `predict_lensing_identity`,
  `predict_isocurvature_identity`, `predict_bbn_background`,
  `predict_ppn_identity`, `predict_gw_identity`,
  `predict_binary_identity`, `predict_shadow_scale`,
  `minkowski_metric`, `lorentzian_response_projector`,
  `predict_bianchi_response`,
  `gate_response`, `predict_gate_response`, `predict_linear_stability`,
  and `predict_particle_mesh`.

FITS/WCS coordinate reprojection remains an adapter responsibility; all proxy
map transformations, feature construction, fitting, and forward predictions
are model responsibilities. Statistical nested-scale selection remains in the
cluster validator and only selects among predictions already produced by the
model.

## Evidence labels

- `released_data_forward_calculation`: applies the built-in equations directly
  to pinned public measurements.
- `released_data_model_equivalent`: evaluates released data with a calculation
  that is observationally identical to the pressureless reference model.
- `released_data_proxy_map_calculation`: retrains and validates the cluster-map
  morphology projection with whole-cluster splits.
- `embedded_published_summary_calculation`: transparent arithmetic from cited
  published central values; it is not represented as a raw-data likelihood.
- `equation_identity`: follows by construction from the stated equations.
- `synthetic_numerical_check`: deterministic consistency test, not new
  observational evidence.

The SPARC row extracts the Chae et al. external-field table locally and reports
the requested fixed algebraic simple-MOND comparison.
The cluster rows use light and X-ray morphology proxies, not calibrated
stress-energy. The Sgr A* row is explicitly a published mass-distance scale
calculation, not a visibility-data refit.

## Install and test

From the repository root:

```bash
python3 -m pip install -r darkmatter/review_validation/requirements.txt
python3 -m unittest darkmatter.review_validation.test_review_validation
```

The focused tests check the 22+1 registry, all 55 checksum pins, all 23 exact scalar
values, key SPARC/Bullet/DF4 comparison details, formula ownership, explicit
MOND/NFW separation, sentinel flow, missing-data reports, flat isolation
boundaries, and the AST audit for unnamed numeric literals.

## Run with existing raw data

```bash
python3 -m darkmatter.review_validation.run_validation \
  --data-dir darkmatter/data/raw \
  --output-dir darkmatter/review_validation/runtime_output
```

Select a quick subset with `--cases galaxy_dynamics,df4,gate_robustness`.
Missing raw data is reported in `summary.json` and produces exit status 2.

## Download

No download occurs unless `--download` is supplied:

```bash
python3 -m darkmatter.review_validation.run_validation --download
```

The complete allow-list in `sources.py`, including every cluster product, pins
URLs and SHA-256 digests. Downloads go only under `runtime_data/`, first use a
`.part` file, and are moved into place only after checksum verification.
`--force-download` replaces existing files. Every required file is also
hash-verified when an existing `--data-dir` is used.

## Outputs

`runtime_output/summary.json` records parameters, exact coverage, evidence
counts, methods, values, expected values, tolerances, mismatches, and skips. `summary.csv`,
`summary.png`, and one PNG per completed case are also regenerated. Both runtime
directories are ignored locally; source, tests, documentation, and
configuration remain tracked.
