import pytest
from specsim.datatypes.vector import Vector

def test_vector_initialization() -> None:
    v = Vector(1, 2, 3)
    assert len(v) == 3
    assert v.x == 1
    assert v.y == 2
    assert v.z == 3

def test_vector_empty_initialization() -> None:
    v = Vector()
    assert len(v) == 0
    with pytest.raises(IndexError):
        _ = v.x

def test_vector_index_access() -> None:
    v = Vector(4, 5, 6)
    assert v[0] == 4
    assert v[1] == 5
    assert v[2] == 6

def test_vector_setitem() -> None:
    v = Vector(7, 8, 9)
    v[1] = 10
    assert v[1] == 10

def test_vector_iteration() -> None:
    v = Vector(1, 2, 3)
    elements: list[int] = [element for element in v]
    assert elements == [1, 2, 3]

def test_vector_append() -> None:
    v = Vector(1, 2)
    v.append(3)
    assert len(v) == 3
    assert v[2] == 3

def test_vector_arithmetic_operations() -> None:
    v1 = Vector(1, 2, 3)
    v2 = Vector(4, 5, 6)

    v_add : Vector[int] = v1 + v2
    assert v_add.x == 5
    assert v_add.y == 7
    assert v_add.z == 9

    v_sub : Vector[int] = v1 - v2
    assert v_sub.x == -3
    assert v_sub.y == -3
    assert v_sub.z == -3

    v_mul : Vector[int] = v1 * v2
    assert v_mul.x == 4
    assert v_mul.y == 10
    assert v_mul.z == 18

    v_div : Vector[float] = v2 / v1
    assert v_div.x == 4
    assert v_div.y == 2.5
    assert v_div.z == 2

def test_vector_scalar_operations() -> None:
    v = Vector(1, 2, 3)

    v_add : Vector[int] = v + 1
    assert v_add.x == 2
    assert v_add.y == 3
    assert v_add.z == 4

    v_sub : Vector[int] = v - 1
    assert v_sub.x == 0
    assert v_sub.y == 1
    assert v_sub.z == 2

    v_mul : Vector[int] = v * 2
    assert v_mul.x == 2
    assert v_mul.y == 4
    assert v_mul.z == 6

    v_div : Vector[float] = v / 2
    assert v_div.x == 0.5
    assert v_div.y == 1
    assert v_div.z == 1.5

def test_vector_comparison_operations() -> None:
    v1 = Vector(1, 2, 3)
    v2 = Vector(1, 2, 3)
    v3 = Vector(4, 5, 6)
    v4 = Vector(1, 2)

    assert v1 == v2  # Same elements
    assert v1 != v3  # Different elements
    assert v1 != v4  # Different lengths
    assert v1 != [1, 2, 3]  # Different type
    
def test_vector_property_access() -> None:
    v = Vector(1, 2, 3, 4)
    assert v.x == 1
    assert v.y == 2
    assert v.z == 3
    assert v.a == 4

def test_vector_property_setters() -> None:
    v = Vector(1, 2, 3, 4)
    v.x = 10
    v.y = 20
    v.z = 30
    v.a = 40
    assert v.x == 10
    assert v.y == 20
    assert v.z == 30
    assert v.a == 40

def test_vector_type_enforcement() -> None:
    v = Vector(1, 2, 3)
    with pytest.raises(TypeError):
        v.append("string")
    with pytest.raises(TypeError):
        v[0] = "string"
    with pytest.raises(TypeError):
        v.x = "string"

def test_vector_repr_and_str() -> None:
    v = Vector(1, 2, 3)
    assert repr(v) == "Vector(1, 2, 3)"
    assert str(v) == "Vector(1, 2, 3)"