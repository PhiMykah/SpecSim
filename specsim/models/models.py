from .exponential import sim_exponential_1D
from .gaussian import sim_gaussian_1D
from .composite import sim_composite_1D
from ..debug.verbose import errPrint

from enum import Enum
import sys
class Model(Enum):
    """
    Enum Class for Handling Decay Model Functions

    Raises
    ------
    ValueError
        If input for functions does not resolve to an existing model
    """
    EXPONENTIAL = 0, "exp", sim_exponential_1D
    GAUSSIAN = 1, "gauss", sim_gaussian_1D
    COMPOSITE = 2, "comp", sim_composite_1D

    def __str__(self):
        return self.value[1]
    
    def __int__(self):
        return self.value[0]
    
    def function(self):
        return self.value[2]

    @classmethod
    def from_str(cls, label : str):
        if label in ("ex", "exp", "exponential"):
            return cls.EXPONENTIAL
        elif label in ("gaus", "gauss", "gaussian"):
            return cls.GAUSSIAN
        elif label in ("comp", "compos", "composite"):
            return cls.COMPOSITE
        else:
            errPrint(f"Warning: {label} is not a valid model, using exponential")
            return cls.EXPONENTIAL
    
    @classmethod
    def from_filename(cls, filename : str):
        if filename.endswith("exp") or filename.endswith("ex"):
            return cls.EXPONENTIAL
        elif filename.endswith("gaus") or filename.endswith("gauss"):
            return cls.GAUSSIAN
        elif filename.endswith("comp") or filename.endswith("composite"):
            return cls.COMPOSITE
        else:
            errPrint(f"Warning: Filename '{filename}' does not contain a valid model, using exponential")
            return cls.EXPONENTIAL