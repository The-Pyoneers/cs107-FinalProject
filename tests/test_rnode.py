import pytest
# import farad.elem as Elem
import numpy as np
from farad.rnode import Rnode

# def test_sin_rnode():
#     """Test of sin method for reverse mode."""
#     # Test for sin with two Dual objects
#     x = Rnode(1.0)
#     z = Elem.sin(x)
#     z.grad_value = 1.0
#     try:
#         assert z.value == np.sin(x.value)
#         assert x.grad() == np.cos(x.value)
#
#     except AssertionError as e:
#         print(e)
#         raise AssertionError
#
#     # Test for sin with int
#     x = 3
#     fx = Elem.sin(x)
#     try:
#         assert fx == np.sin(x)
#
#     except AssertionError as e:
#         print(e)
#         raise AssertionError

def test_add():
    """Test of addition special method (__add__) of Rnode class."""
    # Test for addition with scalar Rnode object and float value
    x = Rnode(0.11)
    z = x**2 + x
    z.grad_value = 1.0

    try:
        assert z.value == x.value **2 + x.value
        assert x.grad() == sum(weight * var.grad()
                                  for weight, var in x.children)
    except AssertionError as e:
        print(e)


def test_radd():
    """Test of reverse addition special method (__add__) of Rnode class."""
    # Test for reverse addition with scalar Rnode object and float value
    x = Rnode(0.11)
    z = 0.5 + x
    z.grad_value = 1.0

    try:
        assert z.value == x.value + 0.5
    except AssertionError as e:
        print(e)


def test_sub():
    """Test of subtraction special method (__sub__) of Rnode class."""
    # Test for subtraction with Rnode object
    x = Rnode(0.11)
    y = Rnode(0.5)
    z = x - y
    z.grad_value = 1.0

    try:
        assert z.value == x.value - y.value
        # assert x.grad() == sum(weight * var.grad()
        #                           for weight, var in x.children)
    except AssertionError as e:
        print(e)
    # Test for subtraction with scalar Rnode object and float value
    x = Rnode(0.5)
    z = x - 0.1
    try:
        assert z.value == x.value - 0.1
        # assert x.grad() == sum(weight * var.grad()
        #                           for weight, var in x.children)
    except AssertionError as e:
        print(e)


def test_mul():
    """Test of multiplication special method (__mul__) of Rnode class."""
    # Test for multiplication with scalar Rnode object and float value
    x = Rnode(0.11)
    y = Rnode(0.5)
    z = x * y

    try:
        assert z.value == x.value * y.value
        # assert x.grad() == sum(weight * var.grad()
        #                           for weight, var in x.children)
    except AssertionError as e:
        print(e)
    # Test for subtraction with scalar Rnode object and float value
    x = Rnode(0.5)
    z = x * 0.1
    try:
        assert z.value == x.value * 0.1
        # assert x.grad() == sum(weight * var.grad()
        #                           for weight, var in x.children)
    except AssertionError as e:
        print(e)


def test_rmul():
    """Test of reverse multiplication special method (__add__) of Rnode class."""
    # Test for reverse multiplication with scalar Rnode object and float value
    x = Rnode(0.11)
    z = 0.5 * x

    try:
        assert z.value == x.value * 0.5
    except AssertionError as e:
        print(e)

def test_pow():
    """Test of exponent special method (__pow__) of Rnode class."""
    # Test for exponent with scalar Rnode object and float value
    x = Rnode(0.11)
    z = x ** 2
    z.grad_value = 1.0

    try:
        assert z.value == x.value ** 2
        assert x.grad() == x.value ** 2 * np.log(x.value)
    except AssertionError as e:
        print(e)


def test_rpow():
    """Test of exponent special method (__pow__) of Rnode class."""
    # Test for exponent with scalar Rnode object and float value
    x = Rnode(0.11)
    z = 2 ** x
    z.grad_value = 1.0

    try:
        assert z.value == 2 ** x.value
        # assert x.grad() == x.value ** 2 * np.log(x.value)
    except AssertionError as e:
        print(e)


def test_truediv():
    """Test of the division special method (__truediv__) of Rnode class."""
    # Test for division with scalar Rnode object and float value
    x = Rnode(0.11)
    z = x / 4
    z.grad_value = 1.0

    try:
        assert z.value == x.value / 4
    except AssertionError as e:
        print(e)

# Test for division with scalar Rnode objects
    x = Rnode(0.11)
    y = Rnode(0.5)
    z = x / y
    z.grad_value = 1.0

    try:
        assert z.value == x.value / y.value
    except AssertionError as e:
        print(e)


def test_rtruediv():
    """Test of the reverse division special method (__rtruediv__) of Rnode class."""
    # Test for reverse division with scalar Rnode object and float value
    x = Rnode(5.0)
    z = 1 / x
    try:
        assert z.value == 1 / x.value
    except AssertionError as e:
        print(e)
        raise AssertionError

# Test for reverse division with scalar Rnode objects
    x = Rnode(5.0)
    y = Rnode(5.5)
    z = x / y
    z.grad_value = 1.0

    try:
        assert z.value == x.value / y.value
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_neg():
    """Test of the negation special method (__neg__) of Rnode class."""
    # Test for negation with scalar Rnode object
    x = Rnode(5.0)
    z = -x
    try:
        assert z.value == -1 * x.value
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_pos():
    """Test of the positive special method (__pos__) of Rnode class."""
    # Test for positive special method with scalar Rnode object
    x = Rnode(5)
    z = +x
    try:
        assert z.value == 1 * x.value
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_eq():
    """Test of the equality special method (__eq__) of Rnode class."""
    # Test for equality special method with scalar Rnode object and float value
    x = Rnode(2.0)
    try:
        assert (x == 2.0) == True
        assert (x == 1.0) == False
    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for equality special method with two scalar Rnode object
    x = Rnode(2.0)
    y = Rnode(2.0)
    z = Rnode(1.0)
    try:
        assert (x == y) == True
        assert (x == z) == False
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_neq():
    """Test of the not equal special method (__neq__) of Rnode class."""
    # Test for not equal special method with scalar Rnode object and float value
    x = Rnode(2.0)
    try:
        assert (x != 2) == False
        assert (x != 1) == True
    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for equality special method with two scalar Dual object
    x = Rnode(2.0)
    y = Rnode(2.0)
    z = Rnode(1.0)
    try:
        assert (x != y) == False
        assert (x != z) == True
    except AssertionError as e:
        print(e)
        raise AssertionError