# SPDX-License-Identifier: LGPL-3.0-or-later
"""Submodule containing all the implemented potentials."""

from deepmd.infer import (
    DeepPotential,
)
from deepmd.infer.deep_eval import (
    DeepEval,
)

from .data_modifier import (
    DipoleChargeModifier,
)
from .deep_dipole import (
    DeepDipole,
)
from .deep_dos import (
    DeepDOS,
)
from .deep_polar import (
    DeepGlobalPolar,
    DeepPolar,
)
from .deep_pot import (
    DeepPot,
)
from .deep_wfc import (
    DeepWFC,
)
from .ewald_recp import (
    EwaldRecp,
)
from .model_devi import (
    calc_model_devi,
)

__all__ = [
    "DeepPotential",
    "DeepDipole",
    "DeepEval",
    "DeepGlobalPolar",
    "DeepPolar",
    "DeepPot",
    "DeepDOS",
    "DeepWFC",
    "DipoleChargeModifier",
    "EwaldRecp",
    "calc_model_devi",
]