"""Small shared types and array schemas for adapter inputs and predictions."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Record = dict[str, Any]

# Column layout of one published Bullet Cluster aperture row. The adapter fills
# the rows and the model reads them through these names only.
BULLET_APERTURE_COLUMNS = (
    "plasma_mass_1e12_msun", "plasma_mass_sigma_1e12_msun",
    "galaxy_mass_1e12_msun", "galaxy_mass_sigma_1e12_msun",
    "convergence", "convergence_sigma",
)
(
    _PLASMA_MASS, _PLASMA_MASS_SIGMA, _GALAXY_MASS, _GALAXY_MASS_SIGMA,
    _CONVERGENCE, _CONVERGENCE_SIGMA,
) = range(len(BULLET_APERTURE_COLUMNS))
BULLET_BARYON_MASS_COLUMNS = (
    (_PLASMA_MASS, _PLASMA_MASS_SIGMA),
    (_GALAXY_MASS, _GALAXY_MASS_SIGMA),
)
BULLET_CONVERGENCE_COLUMNS = (_CONVERGENCE, _CONVERGENCE_SIGMA)
BULLET_BCG_POSITION, BULLET_PLASMA_POSITION = 0, 1
