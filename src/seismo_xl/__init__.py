"""seismo-xl: measure frequency shifts in solar-like oscillators.

This package implements the filtered cross-correlation method for computing
variations in p-mode frequencies (δω_ℓ) over time.
"""

from . import config
from . import globalvars
from . import logger
from . import stellarspec
from . import utils

__all__ = ["config", "globalvars", "logger", "stellarspec", "utils"]
__version__ = "0.1.0"