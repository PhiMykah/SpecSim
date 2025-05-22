from types import NotImplementedType

class Phase():
    """
    A class to represent the phase correction parameters.

    Attributes
    ----------
    _p0 : float
        The zero-order phase correction in degrees
    _p1 : float
        The first-order phase correction in degrees
    MIN : float
        The minimum value for phase correction in degrees
    MAX : float
        The maximum value for phase correction in degrees
    """
    MIN : float = -180.0
    MAX : float = 180.0

    def __init__(self, p0: float = 0.0, p1: float = 0.0) -> None:

        """
        Initialize the phase correction parameters.

        Parameters
        ----------
        p0 : float
            The zero-order phase correction in degrees
        p1 : float
            The first-order phase correction in degrees
        """
        self._p0 = p0
        self._p1 = p1


    # ---------------------------------------------------------------------------- #
    #                              Getters and Setters                             #
    # ---------------------------------------------------------------------------- #
    # ------------------------------------ p0 ------------------------------------ #

    @property
    def p0(self) -> float:
        """
        Returns the zero-order phase correction.

        Returns
        -------
        float
            The zero-order phase correction in degrees
        """
        return self._p0
    
    @p0.setter
    def p0(self, new_value) -> None:
        """
        Sets the zero-order phase correction.

        Parameters
        ----------
        new_value : float
            The new zero-order phase correction in degrees
        """
        # Ensure value is float
        if not isinstance(new_value, float):
            raise TypeError("p0 for Phase must be a float value.")
        # Set if the new value is within the range of -180 to 180 degrees
        if new_value < Phase.MIN or new_value > Phase.MAX:
            raise ValueError(f"Phase correction must be between {Phase.MIN} and {Phase.MAX} degrees.")
        self._p0: float = new_value

    # ------------------------------------ p1 ------------------------------------ #

    @property
    def p1(self) -> float:
        """
        Returns the first-order phase correction.

        Returns
        -------
        float
            The first-order phase correction in degrees
        """
        return self._p1
    
    @p1.setter
    def p1(self, new_value) -> None:
        """
        Sets the first-order phase correction.

        Parameters
        ----------
        new_value : float
            The new first-order phase correction in degrees
        """
        # Ensure value is float
        if not isinstance(new_value, float):
            raise TypeError("p0 for Phase must be a float value.")
        # Set if the new value is within the range of -180 to 180 degrees
        if new_value < Phase.MIN or new_value > Phase.MAX:
            raise ValueError(f"Phase correction must be between {Phase.MIN} and {Phase.MAX} degrees.")
        self._p1: float = new_value

    # ---------------------------------------------------------------------------- #
    #                                 Magic Methods                                #
    # ---------------------------------------------------------------------------- #

    def __getitem__(self, key) -> float:
        if key == 0:
            return self._p0
        elif key == 1:
            return self._p1
        else:
            raise IndexError("Index out of range. Valid indices are 0 and 1.")

    def __setitem__(self, key, value) -> None:
        if key == 0:
            # Ensure value is float
            if not isinstance(value, float):
                raise TypeError("p0 for Phase must be a float value.")
            if value < Phase.MIN or value > Phase.MAX:
                raise ValueError(f"Phase correction must be between {Phase.MIN} and {Phase.MAX} degrees.")
            self._p0 : float = value
        elif key == 1:
            # Ensure value is float
            if not isinstance(value, float):
                raise TypeError("p0 for Phase must be a float value.")
            if value < Phase.MIN or value > Phase.MAX:
                raise ValueError(f"Phase correction must be between {Phase.MIN} and {Phase.MAX} degrees.")
            self._p1 : float = value
        else:
            raise IndexError("Index out of range. Valid indices are 0 and 1.")
        
    def __repr__(self) -> str:
        return f"Phase(p0={self.p0:.4f}, p1={self.p1:.4f})"
    
    # ---------------------------- Arithmetic Methods ---------------------------- #

    def __add__(self, other) -> "Phase | NotImplementedType":
        if isinstance(other, Phase):
            return Phase(self.p0 + other.p0, self.p1 + other.p1)
        elif isinstance(other, (int, float)):
            return Phase(self.p0 + other, self.p1 + other)
        else:
            return NotImplemented

    def __sub__(self, other) -> "Phase | NotImplementedType":
        if isinstance(other, Phase):
            return Phase(self.p0 - other.p0, self.p1 - other.p1)
        elif isinstance(other, (int, float)):
            return Phase(self.p0 - other, self.p1 - other)
        else:
            return NotImplemented

    def __mul__(self, other) -> "Phase | NotImplementedType":
        if isinstance(other, Phase):
            return Phase(self.p0 * other.p0, self.p1 * other.p1)
        elif isinstance(other, (int, float)):
            return Phase(self.p0 * other, self.p1 * other)
        else:
            return NotImplemented

    def __truediv__(self, other) -> "Phase | NotImplementedType":
        if isinstance(other, Phase):
            if other.p0 == 0 or other.p1 == 0:
                raise ZeroDivisionError("Division by zero is not allowed.")
            return Phase(self.p0 / other.p0, self.p1 / other.p1)
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero is not allowed.")
            return Phase(self.p0 / other, self.p1 / other)
        else:
            return NotImplemented

    def __neg__(self) -> "Phase":
        return Phase(-self.p0, -self.p1)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Phase):
            return self.p0 == other.p0 and self.p1 == other.p1
        return False