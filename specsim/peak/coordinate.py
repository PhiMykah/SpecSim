# ---------------------------------------------------------------------------- #
#                                  Coordinate                                  #
# ---------------------------------------------------------------------------- #

class Coordinate():
    """
    A class to represent a coordinate in the spectrum space.
    Contains methods for arithmetic operations and type conversion.

    Attributes
    ----------

    _value_in_pts : float
        The value of the coordinate in points
    
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
    """
    def __init__(self, value_in_pts : float, spectral_width : float, coordinate_origin : float, observation_freq : float, total_points : int):
        """
        Initialize the coordinate based on provided spectral information.

        Parameters
        ----------
        value_in_pts : float
            The value in points
        spectral_width : float
            The spectral width in Hz. Also known as the sweep width.
        coordinate_origin : float
            The coordinate system origin in pts
        observation_freq : float
            The observation frequency during data acquisition
        total_points : int
            The total number of points in the coordinate system
        """
        self._value_in_pts = value_in_pts
        self._spectral_width = spectral_width
        self._coordinate_origin = coordinate_origin
        self._obervation_freq = observation_freq
        self._total_points = total_points
        self._value_in_hz = self.to_hz()
        self._value_in_ppm = self.to_ppm()

    # ---------------------------------------------------------------------------- #
    #                              Getters and Setters                             #
    # ---------------------------------------------------------------------------- #

    # ------------------------------------ PTS ----------------------------------- #

    @property
    def pts(self):
        """
        Returns the coordinate value in points (pts).

        Returns
        -------
        float
            The value in points
        """
        return self._value_in_pts
    
    @pts.setter
    def pts(self, new_value):
        """
        Sets the coordinate value in points and updates the other units accordingly.

        Parameters
        ----------
        new_value : float
            The new value to set in points
        """
        self._value_in_pts = new_value
        self.update_units()

    # ------------------------------------ HZ ------------------------------------ #

    @property
    def hz(self):
        """
        Returns the coordinate value in hertz (Hz).

        Returns
        -------
        float
            The value in Hz
        """
        return self._value_in_hz
    
    @hz.setter
    def hz(self, new_value):
        """
        Sets the coordinate value in Hz and updates the other units accordingly.

        Parameters
        ----------
        new_value : float
            The new value to set in Hz
        """
        self._value_in_hz = new_value
        # Convert from hz to points
        delta_between_pts = (-1 * self._spectral_width) / self._total_points
        self._value_in_pts = ((new_value - self._coordinate_origin) / delta_between_pts) + self._total_points
        # Update ppm value
        self._value_in_ppm = self.to_ppm()

    # ------------------------------------ PPM ----------------------------------- #
    
    @property
    def ppm(self):
        """
        Returns the coordinate value in parts per million (ppm).

        Returns
        -------
        float
            The value in ppm
        """
        return self._value_in_ppm
    
    @ppm.setter
    def ppm(self, new_value):
        """
        Sets the coordinate value in ppm and updates the other units accordingly.

        Parameters
        ----------
        new_value : float
            The new value to set in ppm
        """
        self._value_in_ppm = new_value
        # Convert from hz to points
        delta_between_pts = (-1 * self._spectral_width) / self._total_points
        self._value_in_pts = (((new_value * self.observation_freq ) - self._coordinate_origin) / delta_between_pts) + self._total_points
        # Update hz value
        self._value_in_hz = self._value_in_ppm * self._obervation_freq
    
    # ------------------------------ Spectral Width ------------------------------ #

    @property
    def spectral_width(self):
        """
        Returns the spectral width of the spectrum. Also known as the sweep width.

        Returns
        -------
        float
            The spectral width in Hz
        """
        return self._spectral_width
    
    @spectral_width.setter
    def spectral_width(self, new_value):
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
    def origin(self):
        """
        Returns the origin of the coordinate system.

        Returns
        -------
        float
            The origin of the coordinate system in pts
        """
        return self._coordinate_origin
    
    @origin.setter
    def origin(self, new_value):
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
    def observation_freq(self):
        """
        Returns the observation frequency of the spectrum in MegaHertz (MHz).

        Returns
        -------
        float
            The observation frequency in MHz
        """
        return self._obervation_freq
    
    @observation_freq.setter
    def observation_freq(self, new_value):
        """
        Sets the observation frequency of the spectrum and updates the coordinate units accordingly.

        Parameters
        ----------
        new_value : float
            The new observation frequency to set in MHz
        """
        if (self._obervation_freq == 0.0):
            self._obervation_freq = 1.0
        else:
            self._obervation_freq = new_value
        self.update_units()

    # ------------------------------ Total Points ------------------------------- #

    @property
    def total_points(self):
        """
        Returns the total number of points in the spectrum.

        Returns
        -------
        int
            The total number of points in the spectrum
        """
        return self._total_points
    
    @total_points.setter
    def total_points(self, new_value):
        """
        Sets the total number of points in the spectrum and updates the coordinate units accordingly.

        Parameters
        ----------
        new_value : int
            The new total number of points to set
        """
        self._total_points = new_value
        self.update_units()

    # ------------------------------ Data Conversion ----------------------------- #

    def update_units(self):
        """
        Updates the coordinate units based on the current spectral values.

        Returns
        -------
        None
        """
        self._value_in_hz = self.to_hz()
        self._value_in_ppm = self.to_ppm()

    def to_ppm(self):
        """
        Converts the coordinate value from points to parts per million (ppm).

        Returns
        -------
        float
            The coordinate value in ppm
        """
        if (self._spectral_width == 0.0):  
            self._spectral_width  = 1.0
        if (self._obervation_freq == 0.0):
            self._obervation_freq = 1.0

        return self._value_in_hz / self._obervation_freq
    
    def to_hz(self):
        """
        Converts the coordinate value from points to hertz (Hz).

        Returns
        -------
        float
            The coordinate value in Hz
        """
        if (self._spectral_width == 0.0):  
            self._spectral_width  = 1.0
        if (self._obervation_freq == 0.0):
            self._obervation_freq = 1.0

        # Calculate the hz distance between points in the spectrum
        # Value is negative because Hz decreases from left to right
        delta_between_pts = (-1 * self._spectral_width) / self._total_points

        # Calculate the hz value of the current coordinate
        # Hz = origin + (x - size) * Δ
        return (self._coordinate_origin + (self._value_in_pts - self._total_points) * delta_between_pts)

    # ------------------------------ Magic Methods ----------------------------- #

    def __repr__(self):
        return f"Coordinate(pts={self.pts:.4f}, " \
               f"ppm={self.ppm:.4f}, " \
               f"hz={self.hz:.4f}, " \
               f"spectral_width={self.spectral_width:.4f}, " \
               f"origin={self.origin:.4f}, " \
               f"observation_freq={self.observation_freq:.4f}, " \
               f"total_points={self.total_points})"

    def __call__(self, *args, **kwds):
        return self.value
    
    def __float__(self):
        return self.value

    def __int__(self):
        return int(self.value)
    
    def __add__(self, other : float):
        return self._value_in_pts + other

    def __sub__(self, other : float):

        return self._value_in_pts - other

    def __mul__(self, other : float):
        return self._value_in_pts * other
    
    def __truediv__(self, other : float):
        return self._value_in_pts / other
        
    def __floordiv__(self, other : float):
        return self._value_in_pts // other   
    
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                                 Coordinate 2D                                #
# ---------------------------------------------------------------------------- #

class Coordinate2D():
    """
    A class to represent a 2D coordinate in the spectrum space.

    Attributes
    ----------
    _index : int
        The index of the peak coordinate

    _x : Coordinate
        The x-coordinate of the peak
    
    _y : Coordinate
        The y-coordinate of the peak

    _integral_index : int
        The index of the peak integral group
    """
    def __init__(self, index : int, x : Coordinate, y : Coordinate, integral_index : int = 0):
        """
        Initialization of the 2D coordinate.

        Parameters
        ----------
        index : int
            The index of the peak coordinate

        x : Coordinate
            The x-coordinate of the peak
        
        y : Coordinate
            The y-coordinate of the peak

        integral_index : int
            The index of the peak integral group
        """
        self._index = index
        self._integral_index = integral_index
        self._x = x
        self._y = y

    # ---------------------------------------------------------------------------- #
    #                              Getters and Setters                             #
    # ---------------------------------------------------------------------------- #

    # ----------------------------------- Index ---------------------------------- #

    @property
    def index(self):
        """
        Returns the index of the peak coordinate.

        Returns
        -------
        int
            The index of the peak coordinate
        """
        return self._index

    # ------------------------------ Integral Index ------------------------------ #

    @property
    def integral_index(self):
        """Returns the integral index of the peak coordinate.

        Returns
        -------
        int
            The integral index of the peak coordinate
        """
        return self._integral_index
    
    # ------------------------------------- x ------------------------------------ #

    @property
    def x(self):
        """
        Returns the x-coordinate of the peak.

        Returns
        -------
        Coordinate
            The x-coordinate of the peak
        """
        return self._x
    
    @x.setter
    def x(self, new_coordinate):
        """
        Sets the x-coordinate of the peak

        Parameters
        ----------
        new_coordinate : Coordinate
            The new x-coordinate to set
        """
        self._x = new_coordinate

    # ------------------------------------- y ------------------------------------ #

    @property
    def y(self):
        """
        Returns the y-coordinate of the peak.

        Returns
        -------
        Coordinate
            The y-coordinate of the peak
        """
        return self._y
    
    @y.setter
    def y(self, new_coordinate):
        """
        Sets the y-coordinate of the peak.

        Parameters
        ----------
        new_coordinate : Coordinate
            The new y-coordinate to set
        """
        self._y = new_coordinate

    # ------------------------------ Magic Methods ----------------------------- #

    def __repr__(self):
        return f"Coordinate2D(index={self.index}, " \
               f"integral_index={self.integral_index}, " \
               f"x={self.x}, " \
               f"y={self.y})"

    def __getitem__(self, key):
            if key == 0:
                return self._x
            elif key == 1:
                return self._y
            else:
                raise IndexError("Index out of range. Valid indices are 0 and 1.")

    def __setitem__(self, key, value):
        if key == 0:
            self._x = value
        elif key == 1:
            self._y = value
        else:
            raise IndexError("Index out of range. Valid indices are 0 and 1.")