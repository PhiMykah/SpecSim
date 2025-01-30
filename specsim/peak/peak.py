import numpy as np
from .coordinate import Coordinate
from .coordinate import Coordinate2D

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
    linewidth : tuple[float, float]
        Linewidth of the peak dimensions in Pts
    extra_params : dict[str, float]
        Extra parameter values to define the peak feature
    """
    def __init__(self, position : Coordinate2D, intensity : float, linewidths : tuple[float, float], extra_params = {}):
        self.position = position
        self.intensity = intensity
        self.linewidths = linewidths
        self.extra_params = extra_params
        pass

    def __repr__(self):
        return f"Peak(position={self.position}, intensity={self.intensity}, width={self.linewidths}, extra_params={self.extra_params})"
