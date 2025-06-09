from ..datatypes import Vector, Phase
from math import ulp

class OptimizationParams:
    """
    Container for Optimization Params to use during the spectral optimization.

    Attributes
    ----------
    num_of_dimensions : int
        Number of dimensions of the optimization
    trials : int, optional
        number of trials to perform, by default 100
    step_size : float, optional
        step-size of optimization if the optimization requires, by default 0.1
    initial_decay : Vector[float] | None, optional
        Initial decay values for optimization in Hz for each dimension, by default Vector(2.0, 0.0, ...)
    initial_phase : Vector[Phase] | None, optional
        Initial phase values for optimization for each dimension, by default Vector(Phase(0.0, 0.0), ...)
    bounds : Vector[tuple[float]] | None, optional
        Upper and lower bounds for the decay in Hz for each dimension, by default Vector((0,100), (0,20), ...)
    amplitude_bounds : tuple[float, float], optional
        Upper and lower bounds for the amplitude in Hz, by default (0.0, 10.0)
    p0_bounds : tuple[float, float], optional
        Upper and lower bounds for the p0 phase in degrees, by default (-180.0, 180.0)
    p1_bounds : tuple[float, float], optional
        Upper and lower bounds for the p1 phase in degrees, by default (-180.0, 180.0)
    initial_weight : float, optional 
        Initial weight value for the optimization
    Raises
    ------
    TypeError
        Raises type error if any of the attributes are not the proper type
    ValueError
        Raises value error if the values are not properly formatted.
    """
    def __init__(self,
            num_of_dimensions : int,
            trials : int = 100,
            step_size : float = 0.1,
            initial_decay : list[Vector[float]] | Vector[float] | None = None,
            initial_phase : list[Vector[Phase]] | Vector[Phase] | None = None,
            bounds : list[Vector[tuple[float, float]]] | Vector[tuple[float, float]] | None = None,
            amplitude_bounds : tuple[float, float] | None = None,
            p0_bounds : tuple[float, float] | None = None,
            p1_bounds : tuple[float, float] | None = None,
            initial_weight : list[Vector[float]] | Vector[float] | None = None) -> None:

        if not isinstance(num_of_dimensions, int):
            raise TypeError("Number of dimensions for optimization parameters must be an integer.")
        if num_of_dimensions < 1 or num_of_dimensions > 4:
            raise ValueError("Number of dimensions must be a number between 1 and 4 inclusive.")
        self._num_of_dimensions: int = num_of_dimensions
        self.trials = trials
        self.step_size = step_size
        self.initial_decay = initial_decay
        self.initial_phase = initial_phase
        self.bounds = bounds
        self.amplitude_bounds = amplitude_bounds
        self.p0_bounds = p0_bounds
        self.p1_bounds = p1_bounds
        self.initial_weight = initial_weight

    # ---------------------------------------------------------------------------- #
    #                              Getters and Setters                             #
    # ---------------------------------------------------------------------------- #

    # ---------------------------------- trials ---------------------------------- #

    @property
    def trials(self) -> int:
        return self._trials
    
    @trials.setter
    def trials(self, value) -> None:
        try:
            self._trials = int(value)
        except (ValueError, TypeError):
            raise ValueError("The 'trials' optimization parameter must be an integer or a numerical value.")
        
    # --------------------------------- step_size -------------------------------- #

    @property
    def step_size(self) -> float:
        return self._step_size
    
    @step_size.setter
    def step_size(self, value) -> None:
        try:
            self._step_size = float(value)
        except ValueError:
            raise ValueError("The 'step_size' optimization parameter must be a numerical value.")
            
    # ------------------------------- initial_decay ------------------------------ #

    @property
    def initial_decay(self) -> list[Vector[float]]:
        return self._initial_decay
    
    @initial_decay.setter
    def initial_decay(self, value) -> None:
        if value is None:
            if self._num_of_dimensions == 1:
                decay_values : list[float] = [2.0]
            else:
                decay_values : list[float] = [2.0] + [0.0] * (self._num_of_dimensions - 1)
            self._initial_decay : list[Vector[float]] = [Vector(decay_values)]
        elif isinstance(value, Vector) and all(isinstance(v, float) for v in value):
            # Assure that value has a length equal to the number of dimensions
            if len(value) != self._num_of_dimensions:
                raise ValueError(f"Vector and number of dimensions mismatch! Vector \
                                 ({len(value)}) must match {self._num_of_dimensions} dimension(s).")
            self._initial_decay : list[Vector[float]] = [value]
        elif isinstance(value, list) and all(isinstance(v, Vector) and all(isinstance(p, float) for p in v) for v in value):
            # Assure that each vector in the list has a length equal to the number of dimensions
            for v in value:
                if len(v) != self._num_of_dimensions:
                    raise ValueError(f"Vector and number of dimensions mismatch! Vector \
                                     ({len(v)}) must match {self._num_of_dimensions} dimension(s).")
            self._initial_decay : list[Vector[float]] = value
        else:
            raise TypeError("initial_decay must be a float Vector list of float Vectors!")
        
    # ------------------------------- initial_phase ------------------------------ #

    @property
    def initial_phase(self) -> list[Vector[Phase]]:
        return self._initial_phase
    
    @initial_phase.setter
    def initial_phase(self, value) -> None:
        if value is None:
            default_phase = Phase(0.0, 0.0)  # Default phase values
            phase_list: list[Phase] = [default_phase] * self._num_of_dimensions
            self._initial_phase : list[Vector[Phase]] = [Vector(phase_list)]
        elif isinstance(value, Vector) and all(isinstance(v, Phase) for v in value):
            # Assure that value has a length equal to the number of dimensions
            if len(value) != self._num_of_dimensions:
                raise ValueError(f"Vector and number of dimensions mismatch! Vector \
                                 ({len(value)}) must match {self._num_of_dimensions} dimension(s).")
            self._initial_phase : list[Vector[Phase]] = [value]
        elif isinstance(value, list) and all(isinstance(v, Vector) and all(isinstance(p, Phase) for p in v) for v in value):
            # Assure that each vector in the list has a length equal to the number of dimensions
            for v in value:
                if len(v) != self._num_of_dimensions:
                    raise ValueError(f"Vector and number of dimensions mismatch! Vector \
                                     ({len(v)}) must match {self._num_of_dimensions} dimension(s).")
            self._initial_phase : list[Vector[Phase]] = value
        else:
            raise TypeError("initial_phase must be a Phase Vector or list of Phase Vectors!")
        
    # ---------------------------------- bounds ---------------------------------- #

    @property
    def bounds(self) -> list[Vector[tuple[float, float]]]:
        return self._bounds

    @bounds.setter
    def bounds(self, value) -> None:
        if value is None:
            if self._num_of_dimensions == 1:
                default_bounds : list[tuple[float, float]] = [(0.0,100.0)]
            if self._num_of_dimensions == 2:
                default_bounds : list[tuple[float, float]] = [(0.0,100.0), (0.0,20.0)]
            else:
                default_bounds : list[tuple[float, float]] = [(0.0,100.0), (0.0,20.0)] + [(0.0,100.0)] * (self._num_of_dimensions - 2)
            self._bounds : list[Vector[tuple[float, float]]] = [Vector(default_bounds)]
        elif isinstance(value, Vector) and all(isinstance(v, tuple) and len(v) == 2 for v in value):
            if not all(isinstance(v[0], float) and isinstance(v[1], float) for v in value):
                raise ValueError("Each tuple in bounds must contain two float values!")
            if not all(v[0] < v[1] for v in value):
                raise ValueError("Each tuple in bounds must have the first value (lower) less than the second value (higher)!")
            if len(value) != self._num_of_dimensions:
                raise ValueError(f"Vector and number of dimensions mismatch! Vector \
                        ({len(value)}) must match {self._num_of_dimensions} dimension(s).")
            self._bounds : list[Vector[tuple[float, float]]] = [value]
        elif isinstance(value, list) and all(isinstance(v, Vector) and all(isinstance(p, tuple) and len(p) == 2 for p in v) for v in value):
            # Assure that each vector in the list has a length equal to the number of dimensions
            for v in value:
                if len(v) != self._num_of_dimensions:
                    raise ValueError(f"Vector and number of dimensions mismatch! Vector \
                                     ({len(v)}) must match {self._num_of_dimensions} dimension(s).")
            self._bounds : list[Vector[tuple[float, float]]] = value
        else:
            raise TypeError("bounds must be a Vector of tuples with two float values each or a list of Vectors!")

    # ---------------------------- amplitude_bounds ----------------------------- #

    @property
    def amplitude_bounds(self) -> tuple[float, float]:
        return self._amplitude_bounds

    @amplitude_bounds.setter
    def amplitude_bounds(self, value) -> None:
        if value is None:
            self._amplitude_bounds : tuple[float, float] = (0.0, 10.0)
        elif isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, float) for v in value):
            if value[0] > value[1]:
                raise ValueError("amplitude bounds must have the first value (lower) less than the second value (higher)!")
            if value[0] == value[1]:
                self._amplitude_bounds : tuple[float,float] = (value[0], value[0] + ulp(1.0))
            else:
                self._amplitude_bounds : tuple[float, float] = value
        else:
            raise TypeError("amplitude_bounds must be a tuple of two float values!")

    # -------------------------------- p0_bounds -------------------------------- #

    @property
    def p0_bounds(self) -> tuple[float, float]:
        return self._p0_bounds

    @p0_bounds.setter
    def p0_bounds(self, value) -> None:
        if value is None:
            self._p0_bounds : tuple[float, float] = (-180.0, 180.0)
        elif isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, float) for v in value):
            if value[0] > value[1]:
                raise ValueError("p0 bounds must have the first value (lower) less than the second value (higher)!")
            if value[0] == value[1]:
                self._p0_bounds : tuple[float,float] = (value[0], value[0] + ulp(1.0))
            else:
                self._p0_bounds : tuple[float, float] = value
        else:
            raise TypeError("p0_bounds must be a tuple of two float values!")

    # -------------------------------- p1_bounds -------------------------------- #

    @property
    def p1_bounds(self) -> tuple[float, float]:
        return self._p1_bounds

    @p1_bounds.setter
    def p1_bounds(self, value) -> None:
        if value is None:
            self._p1_bounds : tuple[float, float] = (-180.0, 180.0)
        elif isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, float) for v in value):
            if value[0] > value[1]:
                raise ValueError("p1 bounds must have the first value (lower) less than the second value (higher)!")
            if value[0] == value[1]:
                self._p1_bounds : tuple[float,float] = (value[0], value[0] + ulp(1.0))
            else:
                self._p1_bounds : tuple[float, float] = value
        else:
            raise TypeError("p1_bounds must be a tuple of two float values!")
    
    # ------------------------------ Initial Weight ------------------------------ #

    @property
    def initial_weight(self) -> list[Vector[float]]:
        return self._initial_weight
    
    @initial_weight.setter
    def initial_weight(self, value) -> None:
        if value is None:
            self._initial_weight : list[Vector[float]] = [Vector([0.5] * self._num_of_dimensions)]
        elif isinstance(value, float):
            self._initial_weight : list[Vector[float]] = [Vector([max(min(value, 1.0),0.0)] * self._num_of_dimensions)]
        elif isinstance(value, Vector) and all(isinstance(v, float) for v in value):
            clamped_values: list[float] = [max(min(v, 1.0), 0.0) for v in value]
            self._initial_weight : list[Vector[float]] = [Vector(clamped_values)]
        elif isinstance(value, list) and all(isinstance(v, Vector) and all(isinstance(f, float) for f in v) for v in value):
            # Assure that each vector in the list has a length equal to the number of dimensions
            for v in value:
                if len(v) != self._num_of_dimensions:
                    raise ValueError(f"Vector and number of dimensions mismatch! Vector \
                                     ({len(v)}) must match {self._num_of_dimensions} dimension(s).")
            clamped_vectors : list[Vector[float]] = []
            for vector in value:
                clamped_values = [max(min(v, 1.0), 0.0) for v in vector]
                clamped_vectors.append(Vector(clamped_values))

            self._initial_weight : list[Vector[float]] = clamped_vectors
        else:
            raise TypeError("initial_weight must be list of float vectors, float Vector, or a float!")
        
    # ---------------------------------------------------------------------------- #
    #                                Magic Functions                               #
    # ---------------------------------------------------------------------------- #

    def __str__(self) -> str:
        return (f"OptimizationParams(num_of_dimensions={self._num_of_dimensions}, "
                f"trials={self._trials}, step_size={self._step_size}, "
                f"initial_decay={self._initial_decay}, initial_phase={self._initial_phase}, "
                f"bounds={self._bounds}, amplitude_bounds={self._amplitude_bounds}, "
                f"p0_bounds={self._p0_bounds}, p1_bounds={self._p1_bounds}, "
                f"initial_weight={self._initial_weight})")

    def __repr__(self) -> str:
        return (f"OptimizationParams(num_of_dimensions={self._num_of_dimensions}, "
                f"trials={self._trials}, step_size={self._step_size}, "
                f"initial_decay={repr(self._initial_decay)}, initial_phase={repr(self._initial_phase)}, "
                f"bounds={repr(self._bounds)}, amplitude_bounds={repr(self._amplitude_bounds)}, "
                f"p0_bounds={repr(self._p0_bounds)}, p1_bounds={repr(self._p1_bounds)}, "
                f"initial_weight={repr(self._initial_weight)})")