"""Pinned public inputs. Paths are relative to the selected raw-data root."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    path: str
    url: str
    sha256: str
    cases: tuple[str, ...]


PLIK = (
    "https://raw.githubusercontent.com/BaeHenryS/MCMC_CMB/"
    "b79888fbe463ba71ebb943a5768d3e2d3e782ba6/"
    "MCMC/data/planck2018_plik_lite"
)
LENS = (
    "https://raw.githubusercontent.com/CobayaSampler/"
    "planck_supp_data_and_covmats/"
    "621f4c5db33b08f4d28527e3459a922476f49618/lensing/2018"
)
LP = "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8"
SDSS = (
    "https://svn.sdss.org/public/data/eboss/DR16cosmo/tags/"
    "v1_0_1/dataveccov/lrg_elg_qso"
)


def _source(path: str, url: str, digest: str, *cases: str) -> Source:
    return Source(path, url, digest, tuple(cases))


SOURCES = [
    _source(
        "sparc/MassModels_Lelli2016c.mrt",
        "https://astroweb.cwru.edu/SPARC/MassModels_Lelli2016c.mrt",
        "9108994b12cc401b94a1768beca61c53ec354779385c9c9cc571049f3043244c",
        "galaxy_dynamics"),
    _source(
        "efe/chae_2021_external_field.pdf",
        "https://arxiv.org/pdf/2109.04745v1",
        "27a014723628da06befb8e65bba4d71be2e05f7bfbe51b1cc4ce2ae4a8e753a6",
        "galaxy_dynamics"),
    _source(
        "df4/vandokkum2019_df4.pdf",
        "https://arxiv.org/pdf/1901.05973v3",
        "84053b4db0a16974cd0915e302e93459fe55d9caa8978b7ae0156c642464e308",
        "df4"),
    _source(
        "jades_dr4/Combined_DR4_external_v1.2.1.fits",
        "https://jades.herts.ac.uk/DR4/Combined_DR4_external_v1.2.1.fits",
        "4b4e7fd4306208ee801406c7ed73fdab895a2e67184af7f639a91e5f9c8ab7d4",
        "jades"),
]

for name, digest in {
    "blmax.dat": "c28ade0fa5270c7e87ba07bdcb68aef8783b132b352bfaa36c04d17694ab4014",
    "blmin.dat": "325b351cbf8f694556bb13e98f285344e8d66811bb8eef18bcdcf1626518719d",
    "bweight.dat": "8afcbd8bad769e2de96bacd80177e6543f96b2b406e6c2da1fd0d26718c9e415",
    "c_matrix_plik_v22.dat": "ad90378c50bd67841764179c90ae6711fa4317c649966ab2b0712143b31e0a32",
    "cl_cmb_plik_v22.dat": "dac0d9d493213e77c940a10a968cf0da3c5730bae60e1356c4cd8bcff96377ff",
}.items():
    SOURCES.append(
        _source(f"planck/plik_lite_v22/{name}", f"{PLIK}/{name}", digest, "planck_spectra")
    )

for name, digest in {
    f"{LP}_bandpowers.dat": "0113871c95b026dbf544c21f3c0cd667bea25ad146dddb93db4189cff660a6f0",
    f"{LP}_CMBmarged_cov.dat": "1ce3e79dbe1e72858eb50ade73b5ba18a8c5f17a6d48c9f8708d3c65d567b2a4",
    f"{LP}_CMBmarged_lensing_fiducial_correction.dat": "26f97ac6c742656cf5f6f8b9fc5c22c6e6384de9c9340b33bbcd47685c3fcb28",
}.items():
    SOURCES.append(
        _source(f"planck/lensing_2018/{name}", f"{LENS}/{name}", digest, "planck_lensing")
    )

WINDOW_HASHES = {
    "windows": [
        "da21c0c323f3d43b896c0b08832801d553812000e82aa4d30581574f1483c5f6",
        "5a610ba5e9a158cdb4bf1b787c241d980d53f4937141fe7602a107fa6b04996f",
        "29cee1f5ec34a643a6c7091489e58995ee0680f2d2a79df5a99275116fcd07cc",
        "deda1ebff7dcf07c7da63e54ceb90b1fb6cc20d52b22b06b37f705583e805f56",
        "15f649bec0438cdab5c9efab0bed6a92ae139bff00c74e0cf99e89217f72a183",
        "c110f460478ad285899b50ed495f11de2c0f8c7e67f11f4eaabd0d1506e523ca",
        "7225217ea62d9909cbca9adb4b7351072040f925f462a7f47f2b6c262c12fd76",
        "9952b5112111eb998078f7387e01abf139d4b720e17b54097167312f8e2d8d0e",
        "6389ed1301c686cfe4d4d9b237ff63363423918657163fc59bd806909e191db6",
    ],
    "correction_windows": [
        "f60906b71823351552bb2b6b2b90d0d42e0bff5ce74a1f10f153ac92fdc00e7f",
        "2944563cb8ceec52716677b8b9b1a5183f068a584bc8c61190d0f3479871b533",
        "061c3f49d004b66e9466b73cacf7408942d6a0312052e7c279f64a6db3414cfc",
        "521d8b11823c631ea3ffca4e3259491a896b2a10a21952b67264ad957bc6a182",
        "44010273e787013d3bea3bcdfc692aa5c25833c680dcdd6ec9efe6cda93d25a8",
        "3e82460f4b4db176166dc5990917e04adfd3e481cfe6d037fd7c11522eee9221",
        "38ab12bdcdfb5cd89f63bd05a59106f795244d2dd814df8f9528378676cc3986",
        "ea4c7a29d85bfa4b6e529b3a75708b803dc8ce622f705764c0b7659630cc7fd5",
        "37e5356ec9640337d6d2368c7837076bccbd7508bb81a517248d96549fbf4b67",
    ],
}
for directory, hashes in WINDOW_HASHES.items():
    remote = (
        f"{LP}_window"
        if directory == "windows"
        else f"{LP}_CMBmarged_lens_delta_window"
    )
    for index, digest in enumerate(hashes, 1):
        SOURCES.append(
            _source(
                f"planck/lensing_2018/{directory}/window{index}.dat",
                f"{LENS}/{remote}/window{index}.dat",
                digest,
                "planck_lensing"),
        )

SDSS_FILES = {
    "ELG_xi/Covariance_ELGxi_NGCSGC_0.7z1.1_modified2PCF_prerecon_0.6z1.1_standard2PCF_postrecon.txt": "a23faa8d09dad0ae2f67eb216ac54eb08b64274268ee495c45ad37f75d91b70a",
    "ELG_xi/Data_ELGxi_NGCSGC_0.6z1.1_standard2PCF_postrecon.txt": "7295a62aa4497eddb8b7561903a88b00fc1d2e00ef80b77c4251bcefe5c87a89",
    "ELG_xi/Model_BAO_ELGxi_NGCSGC_0.6z1.1_standard2PCF_postrecon_fitmodified2PCFpre_standard2PCFpost.txt": "12ee09b7db95b2e6d8ddf873ad1999256853d11a35ecc64de8b961dcf34f039c",
    "LRG_xi/Covariance_LRGxi_NGCSGC_0.6z1.0_postrecon.txt": "f21f56b14cac5874f0e981603a09ba9c3265e22c6b75793df253116b228f57da",
    "LRG_xi/Data_LRGxi_NGCSGC_0.6z1.0_postrecon.txt": "0f092da82a8c2141b333a1be111ce0847a7bce24d89418f4a3d02fc03cd2a694",
    "LRG_xi/Model_BAO_LRGxi_NGCSGC_0.6z1.0_postrecon.txt": "2f07f3dd9482e870ea1be4c943b5f5f2a99c05e00d6db395a6463372d8a98763",
    "QSO_xi/Covariance_QSOxi_NGCSGC_0.8z2.2_prerecon.txt": "87e262087e76ee033a61ac28e6b3b86a6788d26c472f7527c8b4a8326a1da78f",
    "QSO_xi/Data_QSOxi_NGCSGC_0.8z2.2_prerecon.txt": "c6fa6e1d22b4563d55ab94ca2386d0dfa44cca070aa931d6275868c5c3b06fa1",
    "QSO_xi/Model_QSOxi_NGCSGC_0.8z2.2_prerecon.txt": "b7de4e7457f0cbe41305d395c84159dc8afb7f0aad13904cc629eac83048cec1",
}
for path, digest in SDSS_FILES.items():
    SOURCES.append(
        _source(f"sdss/{path}", f"{SDSS}/{path}", digest, "sdss_eboss")
    )


CLUSTER_CASES = (
    "held_cluster_map_prediction",
    "hff_lens_robustness",
    "abell520_morphology",
    "retrained_cluster_closure",
)
FRONTIER = "https://archive.stsci.edu/pub/hlsp/frontier"
# Chandra observation identifier position inside "acisf<obsid>N<version>...".
OBSERVATION_ID_START, OBSERVATION_ID_END = 5, 10
CDAFTP_DIRECTORY_MODULUS = 10
# system, HST detector, filter, Chandra identifier, then galaxy/Merten/Sharon/X-ray pins
HFF_SPECS = (
    ("abell2744", "acs-60mas-selfcal", "f814w", "07915N003", (
        "56ea591cb58283071f0493ef413cd0621d877fe22c13b08ef615cae8ee4acf02",
        "553468ceb4c7dc104a856db16a5f67d16a269dd0fb309c0cc3bd4dff253fe44a",
        "9e79d81186fe8d7e77abc46eb06c83e8312c84f1dea1df3ae331520640796545",
        "1b534f8fd189f8d5a36166b13d13e8d34dc817fe5584d26fb8a6505b6ea5a749",
    )),
    ("macs0416", "wfc3-60mas-bkgdcor", "f160w", "16236N002", (
        "8c9ba92c04a035e91d4622ba252c6c209a987c53fe26111515ddf0630dc5a723",
        "8b2b5d634bac95bb204829604d1efcf78c6f17e6321398b1fed928ff0af65b7b",
        "a6882cceb2ab129abeb63e068e9ee3d42532712382a1b0fa53621028fd626d06",
        "f74f8f271c8f3d52165f3ce21d1d1a78ba4d21fe55044954dc51d4477d9f115a",
    )),
    ("macs0717", "wfc3-60mas-bkgdcor", "f160w", "16235N003", (
        "257b7100bb0832783ef7d6ed0f960346e93150a800eba6755ea706180a13850b",
        "c3bf9b7ad79a4b285f6ea16dcc2892ad243ba55013146e3eeb7e42639d8cb0ba",
        "4dcb81cb70a5b429687566b850490ee9f727cb909b6c769adbcf03927930511c",
        "d216604cd65733ce6bcf19163cf91b232d3e9a419096d606ca011c3a75cb98d4",
    )),
)


def _xray_url(filename: str) -> str:
    obsid = int(filename[OBSERVATION_ID_START:OBSERVATION_ID_END])
    bucket = obsid % CDAFTP_DIRECTORY_MODULUS
    base = f"https://cxc.cfa.harvard.edu/cdaftp/byobsid/{bucket}/{obsid}"
    return f"{base}/primary/{filename}"


for system, detector, band, observation, digests in HFF_SPECS:
    galaxy = f"hlsp_frontier_hst_{detector}_{system}_{band}_v1.0-epoch2_drz.fits.gz"
    merten = f"hlsp_frontier_model_{system}_merten_v1_kappa.fits"
    sharon = f"hlsp_frontier_model_{system}_sharon_v4cor_kappa.fits"
    xray = f"acisf{observation}_full_img2.fits.gz"
    base = f"{FRONTIER}/{system}"
    products = (
        (f"galaxies/{galaxy}", f"{base}/images/hst/v1.0-epoch2/gzipped/{galaxy}"),
        (f"lensing/{merten}", f"{base}/models/merten/v1/{merten}"),
        (f"lensing/{sharon}", f"{base}/models/sharon/v4cor/{sharon}"),
        (f"xray/{xray}", _xray_url(xray)),
    )
    for (path, url), digest in zip(products, digests):
        SOURCES.append(_source(f"mergers/{system}/{path}", url, digest, *CLUSTER_CASES))

A520_GITHUB = (
    "https://raw.githubusercontent.com/austinpeel/a520-glimpse/"
    "1dd39bcd33917f507c668d22a85dc848931acc95/data"
)
A520_PRODUCTS = (
    ("galaxies/hlsp_relics_hst_acs-wfc3ir_abell520_multi_v1_cat.txt",
     "https://archive.stsci.edu/missions/hlsp/relics/abell520/catalogs/"
     "hlsp_relics_hst_acs-wfc3ir_abell520_multi_v1_cat.txt",
     "b1b0472743b34e74f441e63c7e715b8d5df3330c57826620151070a6b6e7b428"),
    ("lensing/kappa_c12_lambda3.0.fits",
     f"{A520_GITHUB}/kappa_c12_lambda3.0.fits",
     "a92c0e22195b0f1e544b2f0ced6141d34838be539eced4f8e0b704a5abc73b28"),
    ("lensing/kappa_j14_lambda3.0.fits",
     f"{A520_GITHUB}/kappa_j14_lambda3.0.fits",
     "d9f1c7519663065e17577a6df9c155500e3912e10aac1092a6ef2f37318d23d8"),
    ("xray/acisf09426N003_full_img2.fits.gz",
     _xray_url("acisf09426N003_full_img2.fits.gz"),
     "831020b7c3dc712520bd9a4c5daecb30a71515a2f89ddfe109f449bbcf6b0171"),
)
for path, url, digest in A520_PRODUCTS:
    SOURCES.append(_source(f"mergers/abell520/{path}", url, digest, *CLUSTER_CASES))
