import pytest
from farad.dual import Dual
import numpy as np


def test_add():
    """Test of addition special method (__add__) of Dual class."""
    # Test for addition with scalar Dual object and float value
    x = Dual(2)
    fx = x + 3.5
    try:
        assert fx.val == 5.5
        assert fx.der == 1.0
    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for addition with two scalar Dual object
    x = Dual(2.0)
    y = Dual(1.0)
    fx = x + y
    try:
        assert fx.val == 3.0
        assert fx.der == 2.0
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_radd():
    """Test of reverse addition special method (__radd__) of Dual class."""
    # Test for reverse addition with scalar Dual object and float value
    x = Dual(1.5)
    fx = 1.5 + x
    try:
        assert fx.val == 3.0
        assert fx.der == 1.0
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_sub():
    """Test of subtraction special method (__sub__) of Dual class."""
    # Test for subtraction with scalar Dual object and float value
    x = Dual(5)
    fx = x - 0.5
    try:
        assert fx.val == 4.5
        assert fx.der == 0.5
    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for subtraction with two scalar Dual object
    x = Dual(2.0)
    y = Dual(1.0)
    fx = x - y
    try:
        assert fx.val == 1.0
        assert fx.der == 0
    except AssertionError as e:
        print(e)
        raise AssertionError 


def test_rsub():
    """Test of reverse subtraction special method (__rsub__) of Dual class."""
    # Test for reverse subtraction with scalar Dual object and float value
    x = Dual(5)
    fx = 5.5 - x
    try:
        assert fx.val == 0.5
        assert fx.der == 4.5
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_mul():
    """Test of multiplication special method (__mul__) of Dual class."""
    # Test for multiplication with scalar Dual object and float value
    x = Dual(5)
    fx = x * 0.5
    try:
        assert fx.val == 2.5
        assert fx.der == 0.5
    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for multiplication with two scalar Dual object
    x = Dual(2.0)
    y = Dual(1.0)
    fx = x * y
    try:
        assert fx.val == 2.0
        assert fx.der == 3.0
    except AssertionError as e:
        print(e)
        raise AssertionError 


def test_rmul():
    """Test of reverse multiplication special method (__rmul__) of Dual class."""
    # Test for reverse multiplication with scalar Dual object and float value
    x = Dual(5)
    fx = 0.5 * x
    try:
        assert fx.val == 2.5
        assert fx.der == 0.5
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_truediv():
    """Test of the division special method (__truediv__) of Dual class."""
    # Test for division with scalar Dual object and float value
    x = Dual(5)
    fx = x / 2
    try:
        assert fx.val == 2.5
        assert fx.der == 0.5
    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for division with two scalar Dual object
    x = Dual(2.0)
    y = Dual(1.0)
    fx = x / y
    try:
        assert fx.val == 2.0
        assert fx.der == -1.0
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_rtruediv():
    """Test of the reverse division special method (__rtruediv__) of Dual class."""
    # Test for reverse division with scalar Dual object and float value
    x = Dual(5)
    fx = 1 / x
    try:
        assert fx.val == 0.2
        assert fx.der == -25.0
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_neg():
    """Test of the negation special method (__neg__) of Dual class."""
    # Test for negation with scalar Dual object
    x = Dual(5)
    fx = -x
    try:
        assert fx.val == -5.0
        assert fx.der == -1.0
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_pos():
    """Test of the positive special method (__pos__) of Dual class."""
    # Test for positive special method with scalar Dual object
    x = Dual(5)
    fx = +x
    try:
        assert fx.val == 5.0
        assert fx.der == 1.0
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_pow():
    """Test of the power special method (__pow__) of Dual class."""
    # Test for power special method with scalar Dual object and float value
    x = Dual(2)
    fx = x ** 2
    try:
        assert fx.val == 4.0
        assert fx.der == 4.0
    except AssertionError as e:
        print(e)
        raise AssertionError 

    # Test for power special method with two scalar Dual object
    x = Dual(2)
    fx = x ** x
    try:
        assert fx.val == 4.0
        assert fx.der == pytest.approx(6.77, 0.001)
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_rpow():
    """Test of the reverse power special method (__rpow__) of Dual class."""
    # Test for reverse power special method with scalar Dual object and float value
    x = Dual(2)
    fx = 2 ** x
    try:
        assert fx.val == 4.0
        assert fx.der == pytest.approx(2.77, 0.001)
    except AssertionError as e:
        print(e)
        raise AssertionError 


def test_eq():
    """Test of the equality special method (__eq__) of Dual class."""
    # Test for equality special method with scalar Dual object and float value
    x = Dual(2)
    try:
        assert (x == 2) == True
        assert (x == 1) == False
    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for equality special method with two scalar Dual object
    x = Dual(2, [1,0])
    y = Dual(2, [1,0])
    z = Dual(2, [0,1])
    try:
        assert (x == y) == True
        assert (x == z) == False
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_neq():
    """Test of the not equal special method (__neq__) of Dual class."""
    # Test for not equal special method with scalar Dual object and float value
    x = Dual(2)
    try:
        assert (x != 2) == False
        assert (x != 1) == True
    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for equality special method with two scalar Dual object
    x = Dual(2, [1,0])
    y = Dual(2, [1,0])
    z = Dual(2, [0,1])
    try:
        assert (x != y) == False
        assert (x != z) == True
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_lt():
    """Test of the less than special method (__lt__) of Dual class."""
    # Test for less than special method with scalar Dual object and float value
    x = Dual(2)
    try:
        assert (x < 3) == True
        assert (x < 1) == False
    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for less than special method with two scalar Dual object
    a = Dual(2, [1,0])
    b = Dual(2, [1,0])
    c = Dual(2, [0,1])
    d = Dual(1, [0,1])
    try:
        assert (a < b) == False
        assert (a < c) == False
        assert (d < a) == True
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_le():
    """Test of the less than or equal to special method (__le__) of Dual class."""
    # Test for less than or equal to special method with scalar Dual object and float value
    x = Dual(2)
    try:
        assert (x <= 3) == True
        assert (x <= 2) == True
        assert (x <= 1) == False
    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for less than or equal to special method with two scalar Dual object
    a = Dual(2, [1,0])
    b = Dual(2, [1,0])
    c = Dual(2, [0,1])
    d = Dual(1, [0,1])
    try:
        assert (a <= b) == True
        assert (a <= c) == True
        assert (a <= d) == False
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_gt():
    """Test of the greater than special method (__gt__) of Dual class."""
    # Test for greater than special method with scalar Dual object and float value
    x = Dual(2)
    try:
        assert (x > 3) == False
        assert (x > 1) == True
    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for greater than special method with two scalar Dual object
    a = Dual(2, [1,0])
    b = Dual(2, [1,0])
    c = Dual(2, [0,1])
    d = Dual(1, [0,1])
    try:
        assert (a > b) == False
        assert (a > c) == False
        assert (a > d) == True
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_ge():
    """Test of the greater than or equal to special method (__ge__) of Dual class."""
    # Test for greater than or equal to special method with scalar Dual object and float value
    x = Dual(2)
    try:
        assert (x >= 3) == False
        assert (x >= 1) == True
    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for greater than or equal to special method with two scalar Dual object
    a = Dual(2, [1,0])
    b = Dual(2, [1,0])
    c = Dual(2, [0,1])
    d = Dual(1, [0,1])
    try:
        assert (a >= b) == True
        assert (a >= c) == True
        assert (d >= a) == False
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_repr():
    """Test of the representation special method (__repr__) of Dual class."""
    # Test for representation special method with scalar Dual objects
    x = Dual(2)
    y = Dual(2, [0,1])
    try:
        assert repr(x) == 'Dual(2,1)'
        assert repr(y) == 'Dual(2,[0, 1])' 
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_str():
    """Test of the string special method (__str__) of Dual class."""
    # Test for string special method with scalar Dual objects
    x = Dual(2)
    y = Dual(2, [0,1])
    try:
        assert str(x) == 'Forward-mode Dual Object ( Values: 2, Derivatives: 1 )'
        assert str(y) == 'Forward-mode Dual Object ( Values: 2, Derivatives: [0, 1] )'
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_len():
    """Test of the length special method (__len__) of Dual class."""
    # Test for string special method with scalar Dual objects
    x = Dual(2)
    y = Dual(2, [0,1])
    try:
        assert len(x) == 1
        assert len(y) == 1
    except AssertionError as e:
        print(e)
        raise AssertionError
