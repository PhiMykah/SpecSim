from typing import Any
from ..datatypes import Vector, Phase, PointUnits

class Peak():
    """
    A class to represent a peak based on coordinates and other information.
    Contains methods for handling and modifying peak data.

    Attributes
    ----------
    position : Vector[PointUnits]
        Coordinates of peak in the spectrum.
    intensity : float
        Intensity of the peak.
    num_of_dimensions : int
        Number of dimensions of the peak
    linewidth : Vector[PointUnits] | list[Vector[PointUnits]] 
        Linewidth(s) of the peak in each dimension
    phase: Vector[Phase] | list[Vector[Phase]] | None
        List of phases for the peak in each dimension, by default None
    weights : list[Vector[float]] | None
        Weight of the Gaussian component in the composite signal, by default None
    extra_params : dict[str, float]
        Extra parameter values to define the peak feature
    """
    def __init__(self, position: Vector[PointUnits], intensity: float, 
                 num_of_dimensions: int,
                 linewidth: Vector[PointUnits] | list[Vector[PointUnits]], 
                 phase: Vector[Phase] | list[Vector[Phase]] | None = None, 
                 weights: Vector[float] | list[Vector[float]] | None = None, 
                 **extra_params: dict[str, Any]) -> None:
        self._num_of_dimensions : int = num_of_dimensions
        self.position = position
        self.intensity = intensity
        self.linewidth = linewidth
        self.phase = phase
        self.weights = weights
        self.extra_params : dict[str, dict[str, Any]] = extra_params

    # ---------------------------------------------------------------------------- #
    #                               Helper Functions                               #
    # ---------------------------------------------------------------------------- #
    
    def expand_linewidth(self, count: int) -> None:
        """
        Expands the linewidth list by adding copies of the first linewidth element 
        until the list contains the specified number of elements.

        Parameters
        ----------
        count : int
            The desired number of elements in the linewidth list.
        """
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Count must be a positive integer.")
        
        if isinstance(self.linewidth, list) and len(self.linewidth) > 0:
            while len(self.linewidth) < count:
                self.linewidth.append(self.linewidth[0])
        else:
            raise ValueError("Linewidth must be a non-empty list to expand.")
    
    def expand_phase(self, count: int) -> None:
        """
        Expands the phase list by adding copies of the first phase element 
        until the list contains the specified number of elements.

        Parameters
        ----------
        count : int
            The desired number of elements in the phase list.
        """
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Count must be a positive integer.")
        
        if isinstance(self.phase, list) and len(self.phase) > 0:
            while len(self.phase) < count:
                self.phase.append(self.phase[0])
        else:
            raise ValueError("Phase must be a non-empty list to expand.")
        
    # ---------------------------------------------------------------------------- #
    #                              Getters and Setters                             #
    # ---------------------------------------------------------------------------- #

    # --------------------------------- position --------------------------------- #

    @property
    def position(self) -> Vector[PointUnits]:
        return self._position
    
    @position.setter
    def position(self, value) -> None:
        if isinstance(value, Vector) and all(isinstance(v, PointUnits) for v in value):
            # Assure that value has a length equal to the number of dimensions
            if len(value) != self._num_of_dimensions:
                raise ValueError(f"Vector and number of dimensions mismatch! Vector \
                                 ({len(value)}) must match {self._num_of_dimensions} dimension(s).")
            self._position : Vector[PointUnits] = value
        else:
            raise TypeError("Peak position must be a PointUnits Vector.")
        
    # --------------------------------- intensity -------------------------------- #

    @property
    def intensity(self) -> float:
        return self._intensity
    
    @intensity.setter
    def intensity(self, value) -> None:
        if isinstance(value, (int, float)):
            self._intensity : float = value
        else:
            raise TypeError("Peak intensity must be a number.")
        
    # --------------------------------- linewidth -------------------------------- #

    @property
    def linewidth(self) -> list[Vector[PointUnits]]:
        return self._linewidth

    @linewidth.setter
    def linewidth(self, value) -> None:
        if isinstance(value, Vector) and all(isinstance(v, PointUnits) for v in value):
            # Assure that value has a length equal to the number of dimensions
            if len(value) != self._num_of_dimensions:
                raise ValueError(f"Position and number of dimensions mismatch! Vector \
                                 ({len(value)}) must match {self._num_of_dimensions} dimension(s).")
            self._linewidth : list[Vector[PointUnits]] = [value]
        elif isinstance(value, list) and all(isinstance(v, Vector) and all(isinstance(p, PointUnits) for p in v) for v in value):
            # Assure that each vector in the list has a length equal to the number of dimensions
            for v in value:
                if len(v) != self._num_of_dimensions:
                    raise ValueError(f"Vector and number of dimensions mismatch! Vector \
                                     ({len(v)}) must match {self._num_of_dimensions} dimension(s).")
            self._linewidth : list[Vector[PointUnits]] = value
        else:
            raise TypeError("linewidth must be a PointUnits Vector or a list of PointUnits Vectors.")
        
    # --------------------------------- phases ----------------------------------- #

    @property
    def phase(self) -> list[Vector[Phase]]:
        return self._phase
    
    @phase.setter
    def phase(self, value) -> None:
        if value is None:
            default_phase = Phase(0.0, 0.0)  # Default phase values
            phase_list : list[Phase] = [default_phase] * self._num_of_dimensions
            self._phase : list[Vector[Phase]] = [Vector(phase_list)]
        elif isinstance(value, Vector) and all(isinstance(v, Phase) for v in value):
            # Assure that value has a length equal to the number of dimensions
            if len(value) != self._num_of_dimensions:
                raise ValueError(f"Vector and number of dimensions mismatch! Vector \
                                 ({len(value)}) must match {self._num_of_dimensions} dimension(s).")
            self._phase : list[Vector[Phase]] = [value]
        elif isinstance(value, list) and all(isinstance(v, Vector) and all(isinstance(p, Phase) for p in v) for v in value):
            # Assure that each vector in the list has a length equal to the number of dimensions
            for v in value:
                if len(v) != self._num_of_dimensions:
                    raise ValueError(f"Vector and number of dimensions mismatch! Vector \
                                     ({len(v)}) must match {self._num_of_dimensions} dimension(s).")
            self._phase : list[Vector[Phase]] = value
        else:
            raise TypeError("phase must be a Phase Vector or a list of Phase Vectors.")
        
    # --------------------------------- weights ---------------------------------- #

    @property
    def weights(self) -> list[Vector[float]] | None:
        return self._weights
    
    @weights.setter
    def weights(self, value) -> None:
        if value is None:
            self._weights = None
        elif isinstance(value, Vector) and all(isinstance(v, float) for v in value):
            # Assure that value has a length equal to the number of dimensions
            if len(value) != self._num_of_dimensions:
                raise ValueError(f"Vector and number of dimensions mismatch! Vector \
                                 ({len(value)}) must match {self._num_of_dimensions} dimension(s).")
            self._weights : list[Vector[float]] | None = [value]
        elif isinstance(value, list) and all(isinstance(v, Vector) and all(isinstance(p, float) for p in v) for v in value):
            # Assure that each vector in the list has a length equal to the number of dimensions
            for v in value:
                if len(v) != self._num_of_dimensions:
                    raise ValueError(f"Vector and number of dimensions mismatch! Vector \
                                     ({len(v)}) must match {self._num_of_dimensions} dimension(s).")
            self._weights : list[Vector[float]] | None = value
        else:
            raise TypeError("weights must be a float Vector or a list of float Vectors.")
        
    # ---------------------------------------------------------------------------- #
    #                                 Magic Methods                                #
    # ---------------------------------------------------------------------------- #

    def __repr__(self) -> str:
        return f"Peak(position={self.position}, intensity={self.intensity}, " \
           f"linewidth={self.linewidth}, phase={self.phase}, weights={self.weights}, " \
           f"extra_params={self.extra_params})"
    
    def __str__(self) -> str:
        result = f"Peak(position={self.position}, intensity={self.intensity}, " \
             f"linewidth={self.linewidth}"
        if self.phase is not None:
            result += f", phase={self.phase}"
        if self.weights is not None:
            result += f", weights={self.weights}"
        if self.extra_params:
            result += f", extra_params={self.extra_params}"
        result += ")"
        return result
    
    def __len__(self) -> int:
        return self._num_of_dimensions
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Peak):
            return NotImplemented
        return (
            self.position == other.position and
            self.intensity == other.intensity and
            self.linewidth == other.linewidth and
            self.phase == other.phase and
            self.weights == other.weights and
            self.extra_params == other.extra_params
        )