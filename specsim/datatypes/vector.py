from typing import Generic, Iterator, TypeVar, Any
from types import NotImplementedType

T = TypeVar('T')

class Vector(Generic[T]):
    """
    A container for multi-dimensional data 
    with memory access, arithmetic operations, 
    and more.

    This class is a simple wrapper around a list to provide a more structured
    interface for handling multi-dimensional data. It allows for easy access
    to the elements of the list using attributes like `x`, `y`, `z`, and `a`.

    Attributes
    ----------
    elements : list
        The underlying list that stores the data.
    index : int, optional
        An optional index for the data, default is None.
    """
    def __init__(self, *args : T | list[T], index: int | None = None) -> None:
        if len(args) == 1 and isinstance(args[0], list):
            self.elements: list[Any] = args[0]
            self.datatype : type[list[Any] | type[T]] | None = type(args[0][0])
        elif args is not None:
            self.elements = list(args)
            self.datatype : type[list[Any] | type[T]] | None = type(args[0])
        else:
            self.datatype = None
        self.index : int | None = index

    def append(self, value) -> None:
        """
        Appends a value to the vector.

        Parameters
        ----------
        value : any
            The value to append to the vector.
        """
        # Check if the value is of the same type as the existing data.
        if self.datatype is None or len(self.elements) == 0:
            self.datatype = type(value)
        if self.datatype is not None and not isinstance(value, self.datatype):
            raise TypeError(f"Value must be of type {self.datatype.__name__}.")
        self.elements.append(value)

    # ---------------------------------------------------------------------------- #
    #                              Getters and Setters                             #
    # ---------------------------------------------------------------------------- #

    # ----------------------------------- data ----------------------------------- #

    def __getitem__(self, index: int) -> T:
        return self.elements[index]

    def __setitem__(self, index, value) -> None:
        if len(self.elements) == 0:
            raise IndexError(f"Unable to perform element access on an empty vector!")
        elif self.datatype is None:
            self.datatype = type(value)
        elif self.datatype is not None and not isinstance(value, self.datatype):
            raise TypeError(f"Value must be of type {self.datatype.__name__}.")
        self.elements[index] = value

    # ------------------------------------- x ------------------------------------ #

    @property
    def x(self) -> T:
        if len(self.elements) > 0:
            return self.elements[0]
        raise IndexError("x is out of range for this Vector.")

    @x.setter
    def x(self, value) -> None:
        if self.datatype is not None and not isinstance(value, self.datatype):
            raise TypeError(f"Value must be of type {self.datatype.__name__}.")
        if len(self.elements) > 0:
            self.elements[0] = value
        else:
            self.append(value)

    # ------------------------------------- y ------------------------------------ #
    @property
    def y(self) -> T:
        if len(self.elements) > 1:
            return self.elements[1]
        raise IndexError("y is out of range for this Vector.")

    @y.setter
    def y(self, value) -> None:
        if self.datatype is not None and not isinstance(value, self.datatype):
            raise TypeError(f"Value must be of type {self.datatype.__name__}.")
        if len(self.elements) > 1:
            self.elements[1] = value
        elif len(self.elements) == 1:
            self.append(value)
        else:
            raise IndexError("y is out of range for this Vector.")

    # ------------------------------------- z ------------------------------------ #
    @property
    def z(self) -> T:
        if len(self.elements) > 2:
            return self.elements[2]
        raise IndexError("z is out of range for this Vector.")

    @z.setter
    def z(self, value) -> None:
        if self.datatype is not None and not isinstance(value, self.datatype):
            raise TypeError(f"Value must be of type {self.datatype.__name__}.")
        if len(self.elements) > 2:
            self.elements[2] = value
        elif len(self.elements) == 2:
            self.append(value)
        else:
            raise IndexError("z is out of range for this Vector.")

    # ------------------------------------- a ------------------------------------ #
    @property
    def a(self) -> T:
        if len(self.elements) > 3:
            return self.elements[3]
        raise IndexError("a is out of range for this Vector.")

    @a.setter
    def a(self, value) -> None:
        if self.datatype is not None and not isinstance(value, self.datatype):
            raise TypeError(f"Value must be of type {self.datatype.__name__}.")
        if len(self.elements) > 3:
            self.elements[3] = value
        elif len(self.elements) == 3:
            self.append(value)
        else:
            raise IndexError("a is out of range for this Vector.")

    # ---------------------------------------------------------------------------- #
    #                                 Magic Methods                                #
    # ---------------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.elements)
    
    def __iter__(self) -> Iterator[T]:
        return iter(self.elements)
    
    def __repr__(self) -> str:
        return f"Vector({', '.join(map(str, self.elements))})"
    
    def __str__(self) -> str:
        return f"Vector({', '.join(map(str, self.elements))})"
    
    # ---------------------------------------------------------------------------- #
    #                             Arithmetic Operations                            #
    # ---------------------------------------------------------------------------- #

    def __add__(self, other) -> "Vector[list[Any]] | NotImplementedType":
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vectors must be of the same length to add.")
            return Vector([a + b for a, b in zip(self.elements, other.elements)])
        elif isinstance(other, (int, float)):
            return Vector([a + other for a in self.elements])
        else:
            return NotImplemented

    def __sub__(self, other) -> "Vector[list[Any]] | NotImplementedType":
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vectors must be of the same length to subtract.")
            return Vector([a - b for a, b in zip(self.elements, other.elements)])
        elif isinstance(other, (int, float)):
            return Vector([a - other for a in self.elements])
        else:
            return NotImplemented

    def __mul__(self, other) -> "Vector[list[Any]] | NotImplementedType":
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vectors must be of the same length to multiply.")
            return Vector([a * b for a, b in zip(self.elements, other.elements)])
        elif isinstance(other, (int, float)):
            return Vector([a * other for a in self.elements])
        else:
            return NotImplemented

    def __truediv__(self, other) -> "Vector[list[Any]] | NotImplementedType":
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vectors must be of the same length to divide.")
            return Vector([a / b for a, b in zip(self.elements, other.elements)])
        elif isinstance(other, (int, float)):
            return Vector([a / other for a in self.elements])
        else:
            return NotImplemented
        
    # ---------------------------------------------------------------------------- #
    #                             Comparison Operations                            #
    # ---------------------------------------------------------------------------- #

    def __eq__(self, other) -> bool:
        if not isinstance(other, Vector):
            return False
        return len(self) == len(other) and all(a == b for a, b in zip(self.elements, other.elements))
    
    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    

    