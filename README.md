# Causal Baryon Gravity effective framework

This repository implements the public model used by the current Nuclear
Physics B manuscript: one closed-form spherical weak-field acceleration law.
It is a compact, reviewable implementation rather than a general relativistic
completion or a multi-domain theory-validation ledger.

For baryonic acceleration \(g_b(r)\),

```text
g_CBG(r) = g_b(r) * [
  1/2 + sqrt(1/4 + a0/g_b(r))
  + A_mem * (
      erf(r/(sqrt(2)*sigma_h))
      - sqrt(2/pi)*(r/sigma_h)*exp(-r^2/(2*sigma_h^2))
    )
]
```

The reproducible constants are:

| Parameter | Value | Role |
|---|---:|---|
| `a0` | `1.2e-10 m/s^2` | spherical weak-field acceleration scale |
| `A_mem` | `5.361930294906166` | asymptotic memory amplitude |
| `sigma_h` | `150 kpc comoving` | radial memory scale |

The radius supplied to the model must therefore be a **comoving kpc** radius.
An observational adapter is responsible for converting a physical radius with
the dataset's redshift convention before calling the model. The code does not
hide that conversion inside the force law.

## Local setup, test, and run

These commands work from a clean clone of this PR branch and do not require
files from a manuscript repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
python3 -m causal_baryon_gravity summary --output-dir outputs
python3 -m causal_baryon_gravity evaluate \
  --g-b-m-s2 1e-11 \
  --radius-kpc-comoving 150 \
  --radius-kpc-physical 150
```

The summary command writes:

- `outputs/summary.json`: constants, derived adapter values, manuscript
  benchmarks, limitations, and claim boundaries;
- `outputs/summary.csv`: a compact table of the headline values;
- `outputs/headline_summary.png`: headline benchmark visualization;
- `outputs/acceleration_profile.csv` and
  `outputs/acceleration_profile.png`: a directly evaluated example profile.

Expected headline values are SPARC RMSE `22.811405517468 km/s` over 3,389
radii in 175 galaxies; canonical-only RMSE `22.807528591682 km/s`; Bullet
response-to-baryon normalization `6.3619303`, centroid `86.639275 kpc`, and
aperture deviance `2.657774`; cluster-map mean `0.739675`, HFF minimum
`0.674837`, and A520 minimum `0.481154`.

The versioned `causal_baryon_gravity/data/headline_inputs.json` contains the
reduced sufficient statistics used to regenerate these summary values. It is
not represented as raw observational data. The model/profile calculations
themselves use only the equation and constants above.

## Domain adapters and interpretation

- **SPARC:** a rotation-curve adapter converts baryonic components to
  \(g_b(r)\), converts radii to the declared comoving convention, evaluates
  the fixed law, and uses the physical radius (passed separately) to convert
  acceleration back to circular speed. No galaxy-specific model parameter is
  introduced by this package.
- **Bullet Cluster and cluster maps:** the repository reports the manuscript's
  reduced response/centroid/deviance and morphology-adapter summaries. These
  are domain-adapter results, not a universal map-level likelihood.
- **Cosmology:** `omega_b=0.02238`, `omega_resp=0.1200`, and
  `omega_nu=0.06/93.14` give the transfer-adapter response fraction
  `0.8390189009861309`. These numbers are transfer estimates only; they are
  not predictions of a derived baryon-only cosmology.
- **DF2/DF4:** the fixed-law dispersions are `21.113989 km/s` and
  `19.757741 km/s`. They are limitations: less discrepant than a canonical
  abundance-matched NFW comparison, but worse than baryons alone.

## Claim boundary

Causal tensor memory is a proposed completion and interpretation of the
effective radial term. This repository does not contain a completed covariant
action and does not derive PPN parameters, gravitational-wave propagation,
binary dynamics, black-hole solutions, conservation laws, stability results,
or Einstein-Boltzmann evolution. The public, testable claim implemented here
is the spherical weak-field law and the explicitly labelled domain adapters.

## Optional public-data download

The default test and summary paths are offline. To fetch the public SPARC mass
model table separately, with its SHA-256 checked:

```bash
python3 download_public_data.py --output-dir public_data
```

The downloader does not run as part of installation, tests, or summary
generation.
