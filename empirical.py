"""Input adapters for released empirical data products.

Page ranges, header rows, band columns, and sample cuts of every released
product are named module constants here.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from pypdf import PdfReader
from scipy.io import FortranFile

from .records import Record

# Chae et al. (2021) arXiv PDF: zero-based page ranges of the fitted and the
# environmental external-field tables.
CHAE_FITTED_PAGES = range(20, 24)
CHAE_ENVIRONMENT_PAGES = range(24, 27)
# SPARC mass-model machine-readable table header block.
SPARC_HEADER_ROWS = 25
SPARC_COLUMNS = (
    "galaxy", "distance_mpc", "radius_kpc", "v_obs_kms", "e_v_obs_kms",
    "v_gas_kms", "v_disk_kms", "v_bulge_kms", "sb_disk", "sb_bulge",
)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _efe_environment(path: Path) -> Record:
    """Extract all matched Chae et al. (2021) external-field table rows."""
    reader = PdfReader(path)
    fitted: list[dict[str, str | float]] = []
    for page_index in CHAE_FITTED_PAGES:
        text = reader.pages[page_index].extract_text(extraction_mode="layout")
        for line in text.splitlines():
            clean = line.replace("−", "-")
            match = re.match(
                r"^\s*([A-Za-z0-9+_-]+)\s+([PABC])\s+"
                r"(-?\d+\.\d+)\s+(-?\d+\.\d+)"
                r"\+(\d+\.\d+)-([0-9.]+)",
                clean,
            )
            if match:
                fitted.append({"galaxy": match.group(1), "x0_3": float(match.group(3))})
    environment: list[dict[str, str | float]] = []
    for page_index in CHAE_ENVIRONMENT_PAGES:
        text = reader.pages[page_index].extract_text(extraction_mode="layout")
        for line in text.splitlines():
            clean = line.replace("−", "-").replace("±", "+/-")
            match = re.match(
                r"^\s*([A-Za-z0-9+_-]+)\s+"
                r"(-?\d+\.\d+)\s*\+/-\s*(\d+\.\d+)\s+"
                r"(-?\d+\.\d+)\s*\+/-\s*(\d+\.\d+)",
                clean,
            )
            if match:
                environment.append({
                    "galaxy": match.group(1),
                    "log10_eN_max_clustering": float(match.group(2)),
                })
    frame = pd.DataFrame(fitted).merge(pd.DataFrame(environment), on="galaxy")
    return {
        "galaxies": frame.galaxy.to_numpy(dtype=str),
        "fitted_log10_acceleration": frame.x0_3.to_numpy(dtype=float),
        "environmental_log10_acceleration": (
            frame.log10_eN_max_clustering.to_numpy(dtype=float)
        ),
    }


def galaxy_dynamics(data_dir: Path) -> Record:
    """Adapt SPARC rotation curves and Chae external fields without fitting."""
    path = require(data_dir / "sparc/MassModels_Lelli2016c.mrt")
    efe_path = require(data_dir / "efe/chae_2021_external_field.pdf")
    frame = pd.read_csv(
        path, sep=r"\s+", skiprows=SPARC_HEADER_ROWS,
        names=SPARC_COLUMNS, engine="python",
    )
    return {
        "galaxies": frame.galaxy.to_numpy(dtype=str),
        "radius_kpc": frame.radius_kpc.to_numpy(dtype=float),
        "observed_speed_km_s": frame.v_obs_kms.to_numpy(dtype=float),
        "gas_speed_km_s": frame.v_gas_kms.to_numpy(dtype=float),
        "disk_speed_km_s": frame.v_disk_kms.to_numpy(dtype=float),
        "bulge_speed_km_s": frame.v_bulge_kms.to_numpy(dtype=float),
        "external_fields": _efe_environment(efe_path),
        "sources": (
            "sparc/MassModels_Lelli2016c.mrt",
            "efe/chae_2021_external_field.pdf",
        ),
    }


N_TT, N_TE, N_EE = 215, 199, 199
N_TOTAL = N_TT + N_TE + N_EE
PLIK_CALIBRATION_SIGMA = 0.005
PLIK_BANDPOWER_COLUMN = 1


def planck_spectra(data_dir: Path) -> Record:
    """Load Planck binned spectra, windows, covariance, and calibration prior."""
    directory = require(data_dir / "planck/plik_lite_v22")
    table = np.loadtxt(require(directory / "cl_cmb_plik_v22.dat"))
    with FortranFile(require(directory / "c_matrix_plik_v22.dat"), "r") as handle:
        packed = handle.read_reals(dtype=float).reshape((N_TOTAL, N_TOTAL))
    lower = np.tril(packed)
    covariance = lower + np.tril(lower, -1).T
    return {
        "observed": np.asarray(table[:, PLIK_BANDPOWER_COLUMN], dtype=float),
        "covariance": np.asarray(covariance, dtype=float),
        "bin_minimum": np.loadtxt(require(directory / "blmin.dat")).astype(int),
        "bin_maximum": np.loadtxt(require(directory / "blmax.dat")).astype(int),
        "bin_weights": np.loadtxt(require(directory / "bweight.dat")),
        "bin_counts": (N_TT, N_TE, N_EE),
        "calibration_sigma": PLIK_CALIBRATION_SIGMA,
        "source": "planck/plik_lite_v22",
    }


LENS_PREFIX = "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8"
LENSING_WINDOW_COUNT = 9
LENSING_MULTIPOLE_COLUMN, LENSING_WEIGHT_COLUMN = 0, 1
LENSING_BANDPOWER_COLUMN = 4
LENSING_CORRECTION_COLUMN = 1


def _windows(directory: Path) -> tuple[Record, ...]:
    result = []
    for index in range(1, LENSING_WINDOW_COUNT + 1):
        table = np.loadtxt(require(directory / f"window{index}.dat"))
        result.append({
            "multipoles": np.asarray(table[:, LENSING_MULTIPOLE_COLUMN], dtype=int),
            "weights": np.asarray(table[:, LENSING_WEIGHT_COLUMN], dtype=float),
        })
    return tuple(result)


def planck_lensing(data_dir: Path) -> Record:
    """Load released Planck lensing bands, windows, correction, and covariance."""
    directory = require(data_dir / "planck/lensing_2018")
    return {
        "observed": np.loadtxt(
            require(directory / f"{LENS_PREFIX}_bandpowers.dat")
        )[:, LENSING_BANDPOWER_COLUMN],
        "covariance": np.loadtxt(
            require(directory / f"{LENS_PREFIX}_CMBmarged_cov.dat")
        ),
        "primary_windows": _windows(directory / "windows"),
        "correction_windows": _windows(directory / "correction_windows"),
        "fiducial_correction": np.loadtxt(require(
            directory / f"{LENS_PREFIX}_CMBmarged_lensing_fiducial_correction.dat"
        ))[:, LENSING_CORRECTION_COLUMN],
        "source": "planck/lensing_2018",
    }


# tracer, catalogue state, data file, released model file, covariance file, and
# the correlation-function multipoles released for that tracer.
SDSS_SPECS = (
    ("LRG", "post-recon",
     "LRG_xi/Data_LRGxi_NGCSGC_0.6z1.0_postrecon.txt",
     "LRG_xi/Model_BAO_LRGxi_NGCSGC_0.6z1.0_postrecon.txt",
     "LRG_xi/Covariance_LRGxi_NGCSGC_0.6z1.0_postrecon.txt",
     (0, 2)),
    ("QSO", "pre-recon",
     "QSO_xi/Data_QSOxi_NGCSGC_0.8z2.2_prerecon.txt",
     "QSO_xi/Model_QSOxi_NGCSGC_0.8z2.2_prerecon.txt",
     "QSO_xi/Covariance_QSOxi_NGCSGC_0.8z2.2_prerecon.txt",
     (0, 2, 4)),
    ("ELG", "post-recon",
     "ELG_xi/Data_ELGxi_NGCSGC_0.6z1.1_standard2PCF_postrecon.txt",
     "ELG_xi/Model_BAO_ELGxi_NGCSGC_0.6z1.1_standard2PCF_postrecon_fitmodified2PCFpre_standard2PCFpost.txt",
     "ELG_xi/Covariance_ELGxi_NGCSGC_0.7z1.1_modified2PCF_prerecon_0.6z1.1_standard2PCF_postrecon.txt",
     (0,)),
)


def _sdss_covariance(path: Path) -> Record:
    frame = pd.read_csv(
        require(path), sep=r"\s+", comment="#", engine="python",
        names=["ci", "cj", "li", "lj", "si", "sj", "cov"],
    )
    return {
        "left_catalogue": frame.ci.to_numpy(dtype=str),
        "right_catalogue": frame.cj.to_numpy(dtype=str),
        "left_multipole": frame.li.to_numpy(dtype=int),
        "right_multipole": frame.lj.to_numpy(dtype=int),
        "left_separation": frame.si.to_numpy(dtype=float),
        "right_separation": frame.sj.to_numpy(dtype=float),
        "values": frame["cov"].to_numpy(dtype=float),
    }


def sdss_eboss(data_dir: Path) -> Record:
    """Adapt released SDSS/eBOSS measurements, endpoints, and covariances."""
    root = require(data_dir / "sdss")
    tracers = []
    for tracer, catalogue, data_name, model_name, cov_name, multipoles in SDSS_SPECS:
        tracers.append({
            "tracer": tracer,
            "catalogue": catalogue,
            "multipoles": multipoles,
            "observed_table": np.loadtxt(require(root / data_name)),
            "released_endpoint_table": np.loadtxt(require(root / model_name)),
            "covariance": _sdss_covariance(root / cov_name),
        })
    return {"tracers": tuple(tracers), "source": "sdss"}


DF4_TABLE_PAGE = 3
DF4_TRACER_COUNT = 7
DF4_MONTE_CARLO_SAMPLES = 60_000
DF4_RANDOM_SEED = 20250308


def _df4_rows(path: Path) -> pd.DataFrame:
    text = PdfReader(path).pages[DF4_TABLE_PAGE].extract_text(extraction_mode="layout")
    rows = []
    for line in text.splitlines():
        clean = line.replace("−", "-")
        match = re.match(
            r"^\s*(GC-\d+)\s+.*?\s"
            r"(\d+\.\d+)\+(\d+\.\d+)-(\d+\.\d+)\s",
            clean,
        )
        if match:
            radius = re.search(
                r"\s(\d+\.\d+)\s+-\d+\.\d+\s+0\.\d+\s+"
                + re.escape(match.group(2)),
                clean,
            )
            rows.append({
                "id": match.group(1),
                "velocity": float(match.group(2)),
                "upper": float(match.group(3)),
                "lower": float(match.group(4)),
                "radius": float(radius.group(1)) if radius else np.nan,
            })
    if len(rows) != DF4_TRACER_COUNT:
        raise ValueError(f"expected seven DF4 tracers, found {len(rows)}")
    return pd.DataFrame(rows)


def df4(data_dir: Path) -> Record:
    """Extract the seven DF4 tracer velocities and fixed Monte Carlo settings."""
    path = require(data_dir / "df4/vandokkum2019_df4.pdf")
    frame = _df4_rows(path)
    return {
        "tracer_ids": tuple(frame.id.astype(str)),
        "velocities_km_s": frame.velocity.to_numpy(dtype=float),
        "upper_errors_km_s": frame.upper.to_numpy(dtype=float),
        "lower_errors_km_s": frame.lower.to_numpy(dtype=float),
        "radii_kpc": frame.radius.to_numpy(dtype=float),
        "samples": DF4_MONTE_CARLO_SAMPLES,
        "random_seed": DF4_RANDOM_SEED,
        "source": "df4/vandokkum2019_df4.pdf",
    }


JADES_TARGET_SUFFIX = "GS-z14-0"
JADES_SECURE_REDSHIFT_RANGE = (13.5, 14.5)
JADES_SECURE_FLAGS = ("A", "B", "C")
JADES_SURVEY_AREA_ARCMIN2 = 46.0
JADES_STELLAR_MASS_MSUN = 5.0e8
JADES_STAR_FORMATION_EFFICIENCY = 0.20


def jades(data_dir: Path) -> Record:
    """Read the JADES target and package the fixed abundance calculation inputs."""
    catalog = require(data_dir / "jades_dr4/Combined_DR4_external_v1.2.1.fits")
    with fits.open(catalog, memmap=True) as hdul:
        table = hdul["Obs_info"].data
        names = np.asarray(table["z_paper_name"]).astype(str)
        redshifts = np.asarray(table["z_Spec"], dtype=float)
        flags = np.asarray(table["z_Spec_flag"]).astype(str)
    targets = np.flatnonzero(np.char.endswith(names, JADES_TARGET_SUFFIX))
    if len(targets) == 0:
        raise ValueError("GS-z14-0 not found in JADES DR4")
    minimum_redshift, maximum_redshift = JADES_SECURE_REDSHIFT_RANGE
    secure = (
        (redshifts >= minimum_redshift)
        & (redshifts <= maximum_redshift)
        & np.isin(flags, JADES_SECURE_FLAGS)
    )
    return {
        "target_redshift": float(redshifts[targets[0]]),
        "secure_rows": int(np.count_nonzero(secure)),
        "survey_area_arcmin2": JADES_SURVEY_AREA_ARCMIN2,
        "redshift_minimum": minimum_redshift,
        "redshift_maximum": maximum_redshift,
        "stellar_mass_msun": JADES_STELLAR_MASS_MSUN,
        "star_formation_efficiency": JADES_STAR_FORMATION_EFFICIENCY,
        "source": "jades_dr4/Combined_DR4_external_v1.2.1.fits",
    }
