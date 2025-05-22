from types import NotImplementedType

class PointUnits():
    """
    A class to represent a point in n-dimensional space in the spectrum

    Attributes
    ----------
    _value_in_pts : float
        The value of the data unit in points
    _spectral_width : float
        The spectral width of the spectrum in Hz. Also known as the sweep width.
    _coordinate_origin : float
        The origin of the coordinate system (in pts)
    _observation_freq : float
        The observation frequency of the spectrum
    _total_points : int
        The total number of points in the spectrum
    _value_in_ppm : float
        The value of the coordinate in ppm
    _value_in_hz : float
        The value of the coordinate in Hz
    """
    def __init__(self, value_in_pts: float, spectral_width: float, coordinate_origin: float, observation_freq: float, total_points: int):
        """
        Parameters
        ----------
        value_in_pts : float
            The value of the data unit in points
        spectral_width : float
            The spectral width of the spectrum in Hz. Also known as the sweep width.
        coordinate_origin : float
            The origin of the coordinate system (in pts)
        observation_freq : float
            The observation frequency of the spectrum
        total_points : int
            The total number of points in the spectrum
        """
        self._value_in_pts: float = value_in_pts
        self._spectral_width: float = spectral_width
        self._coordinate_origin: float = coordinate_origin
        self._observation_freq: float = observation_freq
        self._total_points: int = total_points
        self._value_in_hz: float = self.value_to_hz()
        self._value_in_ppm: float = self.value_to_ppm()

    # ---------------------------------------------------------------------------- #
    #                              Getters and Setters                             #
    # ---------------------------------------------------------------------------- #

    # ------------------------------------ PTS ----------------------------------- #
    
    @property
    def pts(self) -> float:
        """
        Get the value in points

        Returns
        -------
        float
            The value in points
        """
        return self._value_in_pts
    
    @pts.setter
    def pts(self, value: float) -> None:
        """
        Set the value in points

        Parameters
        ----------
        value : float
            The value in points
        """
        self._value_in_pts = value
        self.update_units()

    # ------------------------------ Spectral Width ------------------------------ #

    @property
    def spectral_width(self) -> float:
        """
        Returns the spectral width of the spectrum. Also known as the sweep width.

        Returns
        -------
        float
            The spectral width in Hz
        """
        return self._spectral_width
    
    @spectral_width.setter
    def spectral_width(self, new_value) -> None:
        """
        Sets the spectral width of the spectrum and updates the coordinate units accordingly.

        Parameters
        ----------
        new_value : float
            The new spectral width to set in Hz
        """
        if (self._spectral_width == 0.0):  
            self._spectral_width  = 1.0
        else:
            self._spectral_width = new_value
        self.update_units()

    # ------------------------------ Coordinate Origin --------------------------- #

    @property
    def origin(self) -> float:
        """
        Returns the origin of the coordinate system.

        Returns
        -------
        float
            The origin of the coordinate system in pts
        """
        return self._coordinate_origin
    
    @origin.setter
    def origin(self, new_value) -> None:
        """
        Sets the origin of the coordinate system and updates the coordinate units accordingly.

        Parameters
        ----------
        new_value : float
            The new origin to set in pts
        """
        self._coordinate_origin = new_value
        self.update_units()

    # --------------------------- Observation Frequency -------------------------- #

    @property
    def observation_freq(self) -> float:
        """
        Returns the observation frequency of the spectrum in MegaHertz (MHz).

        Returns
        -------
        float
            The observation frequency in MHz
        """
        return self._observation_freq
    
    @observation_freq.setter
    def observation_freq(self, new_value) -> None:
        """
        Sets the observation frequency of the spectrum and updates the coordinate units accordingly.

        Parameters
        ----------
        new_value : float
            The new observation frequency to set in MHz
        """
        if (self._observation_freq == 0.0):
            self._observation_freq = 1.0
        else:
            self._observation_freq = new_value
        self.update_units()

    # ------------------------------ Total Points ------------------------------- #

    @property
    def total_points(self) -> int:
        """
        Returns the total number of points in the spectrum.

        Returns
        -------
        int
            The total number of points in the spectrum
        """
        return self._total_points
    
    @total_points.setter
    def total_points(self, new_value) -> None:
        """
        Sets the total number of points in the spectrum and updates the coordinate units accordingly.

        Parameters
        ----------
        new_value : int
            The new total number of points to set
        """
        self._total_points = new_value
        self.update_units()

    # ------------------------------------ PPM ----------------------------------- #
    
    @property
    def ppm(self) -> float:
        """
        Get the value in ppm

        Returns
        -------
        float
            The value in ppm
        """
        return self._value_in_ppm
    
    @ppm.setter
    def ppm(self, value: float) -> None:
        """
        Set the value in ppm

        Parameters
        ----------
        value : float
            The value in ppm
        """
        self._value_in_ppm = value
        # Convert from hz to points
        delta_between_pts = (-1 * self._spectral_width) / self._total_points
        self._value_in_pts = (((value * self.observation_freq ) - self._coordinate_origin) / delta_between_pts) + self._total_points
        # Update hz value
        self._value_in_hz = self._value_in_ppm * self._observation_freq

    # ------------------------------------ HZ ------------------------------------ #

    @property
    def hz(self) -> float:
        """
        Get the value in Hz

        Returns
        -------
        float
            The value in Hz
        """
        return self._value_in_hz
    
    @hz.setter
    def hz(self, value: float) -> None:
        """
        Set the value in Hz

        Parameters
        ----------
        value : float
            The value in Hz
        """
        self._value_in_hz = value
        # Convert from hz to points
        delta_between_pts: float = (-1 * self._spectral_width) / self._total_points
        self._value_in_pts = ((value - self._coordinate_origin) / delta_between_pts) + self._total_points
        # Update ppm value
        self._value_in_ppm = self.value_to_ppm()

    # ---------------------------------------------------------------------------- #
    #                           Data Conversion Functions                          #
    # ---------------------------------------------------------------------------- #

    def update_units(self) -> None:
        """
        Updates the point in n-dimensional space based on the current spectral values.

        Returns
        -------
        None
        """
        self._value_in_hz = self.value_to_hz()
        self._value_in_ppm = self.value_to_ppm()

    def value_to_ppm(self) -> float:
        """
        Converts the point in n-dimensional space from points to parts per million (ppm).

        Returns
        -------
        float
            The coordinate value in ppm
        """
        if (self._spectral_width == 0.0):  
            self._spectral_width  = 1.0
        if (self._observation_freq == 0.0):
            self._observation_freq = 1.0
        if (self._total_points == 0):
            self._total_points = 1

        return self._value_in_hz / self._observation_freq
    
    def value_to_hz(self) -> float:
        """
        Converts the point in n-dimensional space from points to hertz (Hz).

        Returns
        -------
        float
            The coordinate value in Hz
        """
        if (self._spectral_width == 0.0):  
            self._spectral_width  = 1.0
        if (self._observation_freq == 0.0):
            self._observation_freq = 1.0
        if (self._total_points == 0):
            self._total_points = 1

        # Calculate the hz distance between points in the spectrum
        # Value is negative because Hz decreases from left to right
        delta_between_pts: float = (-1 * self._spectral_width) / self._total_points

        # Calculate the hz value of the current coordinate
        # Hz = origin + (x - size) * Δ
        return (self._coordinate_origin + (self._value_in_pts - self._total_points) * delta_between_pts)

    # ---------------------------------------------------------------------------- #
    #                                 Magic Methods                                #
    # ---------------------------------------------------------------------------- #

    def __repr__(self) -> str:
        return f"PointUnits(pts={self.pts:.4f}, " \
               f"ppm={self.ppm:.4f}, " \
               f"hz={self.hz:.4f}, " \
               f"spectral_width={self.spectral_width:.4f}, " \
               f"origin={self.origin:.4f}, " \
               f"observation_freq={self.observation_freq:.4f}, " \
               f"total_points={self.total_points})"

    def __call__(self, *args, **kwds) -> float:
        return self._value_in_pts
    
    def __float__(self) -> float:
        return self._value_in_pts

    def __int__(self) -> int:
        return int(self._value_in_pts)
    
    # --------------------------------- Equality --------------------------------- #

    def __eq__(self, other) -> bool:
        if not isinstance(other, PointUnits):
            return False
        return (
            self._value_in_pts == other._value_in_pts and
            self._spectral_width == other._spectral_width and
            self._coordinate_origin == other._coordinate_origin and
            self._observation_freq == other._observation_freq and
            self._total_points == other._total_points
        )
    
    def __ne__(self, other) -> bool:
        return not self.__eq__(other)
    
    # ---------------------------- Arithmetic Methods ---------------------------- #

    def __add__(self, other) -> "PointUnits | NotImplementedType":
        if isinstance(other, PointUnits):
            return PointUnits(
                self._value_in_pts + other._value_in_pts,
                self._spectral_width,
                self._coordinate_origin,
                self._observation_freq,
                self._total_points
            )
        elif isinstance(other, (int, float)):
            return PointUnits(
                self._value_in_pts + other,
                self._spectral_width,
                self._coordinate_origin,
                self._observation_freq,
                self._total_points
            )
        else:
            return NotImplemented

    def __sub__(self, other) -> "PointUnits | NotImplementedType":
        if isinstance(other, PointUnits):
            return PointUnits(
                self._value_in_pts - other._value_in_pts,
                self._spectral_width,
                self._coordinate_origin,
                self._observation_freq,
                self._total_points
            )
        elif isinstance(other, (int, float)):
            return PointUnits(
                self._value_in_pts - other,
                self._spectral_width,
                self._coordinate_origin,
                self._observation_freq,
                self._total_points
            )
        else:
            return NotImplemented

    def __mul__(self, other) -> "PointUnits | NotImplementedType":
        if isinstance(other, PointUnits):
            return PointUnits(
                self._value_in_pts * other._value_in_pts,
                self._spectral_width,
                self._coordinate_origin,
                self._observation_freq,
                self._total_points
            )
        elif isinstance(other, (int, float)):
            return PointUnits(
                self._value_in_pts * other,
                self._spectral_width,
                self._coordinate_origin,
                self._observation_freq,
                self._total_points
            )
        else:
            return NotImplemented
    def __truediv__(self, other) -> "PointUnits | NotImplementedType":
        if isinstance(other, PointUnits):
            if other._value_in_pts == 0:
                raise ZeroDivisionError("division by zero")
            return PointUnits(
                self._value_in_pts / other._value_in_pts,
                self._spectral_width,
                self._coordinate_origin,
                self._observation_freq,
                self._total_points
            )
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("division by zero")
            return PointUnits(
                self._value_in_pts / other,
                self._spectral_width,
                self._coordinate_origin,
                self._observation_freq,
                self._total_points
            )
        else:
            return NotImplemented

    def __floordiv__(self, other) -> "PointUnits | NotImplementedType":
        if isinstance(other, PointUnits):
            if other._value_in_pts == 0:
                raise ZeroDivisionError("division by zero")
            return PointUnits(
                self._value_in_pts // other._value_in_pts,
                self._spectral_width,
                self._coordinate_origin,
                self._observation_freq,
                self._total_points
            )
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("division by zero")
            return PointUnits(
                self._value_in_pts // other,
                self._spectral_width,
                self._coordinate_origin,
                self._observation_freq,
                self._total_points
            )
        else:
            return NotImplemented