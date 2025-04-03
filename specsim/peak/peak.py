import numpy as np
from .coordinate import Coordinate
from .coordinate import Coordinate2D
from .coordinate import Phase

class Peak():
    """
    A class to represent a peak based on coordinates and other information.
    Contains methods for handling and modifying peak data.

    Attributes
    ----------
    position : Coordinate2D
        Coordinates of peak with coordinate data
    intensity : float
        Intensity of the peak.
    linewidth : Coordinate2D
        Linewidth of the peak dimensions in Pts
    extra_params : dict[str, float]
        Extra parameter values to define the peak feature
    exp_linewidths : Coordinate2D
        Exponential linewidths for the peak dimensions
    gauss_linewidths : Coordinate2D
        Gaussian linewidths for the peak dimensions
    phase_exp : Phase
        Phase for the exponential decay
    phase_gauss : Phase
        Phase for the Gaussian decay
    gauss_weight : float
        Weight of the Gaussian component in the composite signal
    """
    def __init__(self, position: Coordinate2D, intensity: float, linewidths: Coordinate2D, **extra_params):
        self.position = position
        self.intensity = intensity
        self.linewidths = linewidths
        self.extra_params = extra_params if extra_params else {}
        self.exp_linewidths : Coordinate2D = extra_params.get("exp_linewidths", linewidths)
        self.gauss_linewidths : Coordinate2D = extra_params.get("gauss_linewidths", linewidths)
        self.phase_exp = extra_params.get("phase_exp", [Phase(),Phase()])
        self.phase_gauss = extra_params.get("phase_gauss", [Phase(),Phase()])
        self.gauss_weights = extra_params.get("gauss_weight", [1.0, 1.0])
        self._xPhase = Phase()
        self._yPhase = Phase()
        self._phase = [self._xPhase, self._yPhase]
        pass

    @property
    def phase(self):
        return self._phase

    @property
    def xPhase(self):
        return self._xPhase

    @property
    def yPhase(self):
        return self._yPhase

    @phase.setter
    def phase(self, value):
        if not isinstance(value, list) or len(value) != 2 or not all(isinstance(i, Phase) for i in value):
            raise ValueError("Phase must be a list of 2 Phases.")
        self._phase = value
        self._xPhase = value[0]
        self._yPhase = value[1]

    @xPhase.setter
    def xPhase(self, value):
        if isinstance(value, Phase):
            self._xPhase = value
            self._phase[0] = self._xPhase
        else:
            if not isinstance(value, tuple) or len(value) != 2 or not all(isinstance(i, float) for i in value):
                raise ValueError("xPhase must be a tuple of 2 floats.")
            self._xPhase = Phase(p0=value[0], p1=value[1])
            self._phase[0] = self._xPhase

    @yPhase.setter
    def yPhase(self, value):
        if isinstance(value, Phase):
            self._yPhase = value
            self._phase[1] = self._yPhase
        else:
            if not isinstance(value, tuple) or len(value) != 2 or not all(isinstance(i, float) for i in value):
                raise ValueError("yPhase must be a tuple of 2 floats.")
            self._yPhase = Phase(p0=value[0], p1=value[1])
            self._phase[1] = self._yPhase

    def __repr__(self):
        return (f"Peak(position={self.position}, intensity={self.intensity}, width={self.linewidths}, "
                f"extra_params={self.extra_params}, exp_linewidths={self.exp_linewidths}, "
                f"gauss_linewidths={self.gauss_linewidths}, phase_exp={self.phase_exp}, "
                f"phase_gauss={self.phase_gauss}, gauss_weight={self.gauss_weight}, phase={self._phase})")
