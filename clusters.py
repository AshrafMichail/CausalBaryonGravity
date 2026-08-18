"""Input adapters for the released cluster-map products."""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_interp
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

from .empirical import require
from .records import Record


SYSTEMS = ("abell2744", "macs0416", "macs0717")
REDSHIFTS = dict(abell2744=0.308, macs0416=0.396, macs0717=0.545, abell520=0.201)
ARCSEC_PER_DEGREE = 3600.0
BILINEAR_ORDER = 1
HFF_LIGHT_IMAGES = {
    "abell2744": "hlsp_frontier_hst_acs-60mas-selfcal_abell2744_f814w_v1.0-epoch2_drz.fits.gz",
    "macs0416": "hlsp_frontier_hst_wfc3-60mas-bkgdcor_macs0416_f160w_v1.0-epoch2_drz.fits.gz",
    "macs0717": "hlsp_frontier_hst_wfc3-60mas-bkgdcor_macs0717_f160w_v1.0-epoch2_drz.fits.gz",
}
HFF_XRAY_IMAGES = {
    "abell2744": "acisf07915N003_full_img2.fits.gz",
    "macs0416": "acisf16236N002_full_img2.fits.gz",
    "macs0717": "acisf16235N003_full_img2.fits.gz",
}


def _hff_paths(data_dir: Path, system: str) -> dict[str, Path]:
    root = data_dir / "mergers" / system
    return {
        "Merten": require(root / "lensing" / f"hlsp_frontier_model_{system}_merten_v1_kappa.fits"),
        "Sharon": require(root / "lensing" / f"hlsp_frontier_model_{system}_sharon_v4cor_kappa.fits"),
        "light": require(root / "galaxies" / HFF_LIGHT_IMAGES[system]),
        "gas": require(root / "xray" / HFF_XRAY_IMAGES[system]),
    }


def _hff_system(data_dir: Path, system: str) -> Record:
    paths = _hff_paths(data_dir, system)
    with fits.open(paths["Merten"]) as hdul:
        merten = np.squeeze(hdul[0].data).astype(float)
        target_wcs = WCS(hdul[0].header).celestial
    shape = merten.shape
    maps = {"Merten": merten}
    footprints = {"Merten": np.ones(shape)}
    for name in ("Sharon", "light", "gas"):
        with fits.open(paths[name]) as hdul:
            source = np.squeeze(hdul[0].data).astype(float)
            source_wcs = WCS(hdul[0].header).celestial
        maps[name], footprints[name] = reproject_interp(
            (source, source_wcs), target_wcs, shape_out=shape, order="bilinear"
        )
    return {
        "name": system,
        "redshift": REDSHIFTS[system],
        "merten": maps["Merten"],
        "sharon": maps["Sharon"],
        "light": maps["light"],
        "gas": maps["gas"],
        "footprints": tuple(footprints.values()),
        "pixel_arcsec": float(
            np.mean(proj_plane_pixel_scales(target_wcs)) * ARCSEC_PER_DEGREE
        ),
    }


def _abell520(data_dir: Path) -> Record:
    root = data_dir / "mergers/abell520"
    with fits.open(require(root / "lensing/kappa_j14_lambda3.0.fits")) as hdul:
        j14 = np.asarray(hdul[0].data, dtype=float)
        ra = np.asarray(hdul["RA"].data, dtype=float)
        dec = np.asarray(hdul["DEC"].data, dtype=float)
    with fits.open(require(root / "lensing/kappa_c12_lambda3.0.fits")) as hdul:
        c12_values = np.asarray(hdul[0].data, dtype=float)
        c12_ra = np.asarray(hdul["RA"].data, dtype=float)
        c12_dec = np.asarray(hdul["DEC"].data, dtype=float)
    c12 = RegularGridInterpolator(
        (c12_dec[:, 0], c12_ra[0, :]), c12_values,
        bounds_error=False, fill_value=np.nan,
    )(np.column_stack([dec.ravel(), ra.ravel()])).reshape(ra.shape)
    catalog = np.loadtxt(require(
        root / "galaxies/hlsp_relics_hst_acs-wfc3ir_abell520_multi_v1_cat.txt"))
    with fits.open(require(root / "xray/acisf09426N003_full_img2.fits.gz")) as hdul:
        image = np.asarray(hdul[0].data, dtype=float)
        xray_wcs = WCS(hdul[0].header).celestial
    px, py = xray_wcs.world_to_pixel_values(ra, dec)
    footprint = ((px >= 0) & (px <= image.shape[1] - 1)
                 & (py >= 0) & (py <= image.shape[0] - 1))
    gas = np.zeros_like(ra)
    gas[footprint] = map_coordinates(
        image, [py[footprint], px[footprint]], order=BILINEAR_ORDER
    )
    pixel_arcsec = float(
        np.median(np.diff(ra[0]))
        * ARCSEC_PER_DEGREE
        * np.cos(np.deg2rad(np.nanmean(dec)))
    )
    return {
        "name": "abell520",
        "redshift": REDSHIFTS["abell520"],
        "jee_lensing": j14,
        "clowe_lensing": c12,
        "right_ascension": ra,
        "declination": dec,
        "member_catalogue": catalog,
        "gas": gas,
        "gas_footprint": footprint,
        "pixel_arcsec": pixel_arcsec,
    }


def cluster_maps(data_dir: Path) -> Record:
    """Adapt HFF and Abell 520 maps into model-ready cluster records."""
    return {
        "hff_systems": tuple(_hff_system(data_dir, name) for name in SYSTEMS),
        "abell520": _abell520(data_dir),
        "source": "hff and mergers raw FITS/catalog products",
    }
