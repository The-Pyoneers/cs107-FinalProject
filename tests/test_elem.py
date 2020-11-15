"""Test Math functionality for farad package.
"""

import pytest
import numpy as np
import farad.elem as Elem
from farad.dual import Dual


def test_sin():
    """Test of sin method."""
    # Test for sin with two Dual objects
    val1 = Dual(3, [4, 1])
    val2 = Dual(2, [3, 1])
    val = val1 + val2
    z = Elem.sin(val)

    try:
        assert z.val == np.sin(val.val)
        assert z.der[0] == np.cos(val.val) * val.der[0]
        assert z.der[1] == np.cos(val.val) * val.der[1]

    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for sin with int
    x = 3
    fx = Elem.sin(x)
    try:
        assert fx == np.sin(x)

    except AssertionError as e:
        print(e)
        raise AssertionError


def test_cos():
    """Test of cos method."""
    # Test for cos with two Dual objects
    val1 = Dual(3, [4, 1])
    val2 = Dual(2, [3, 1])
    val = val1 + val2
    z = Elem.cos(val)

    try:
        assert z.val == np.cos(val.val)
        assert z.der[0] == -np.sin(val.val) * val.der[0]
        assert z.der[1] == -np.sin(val.val) * val.der[1]

    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for cos with int,
    x = 3
    fx = Elem.cos(x)
    try:
        assert fx == np.cos(x)

    except AssertionError as e:
        print(e)
        raise AssertionError


def test_tan():
    """Test of tan method."""
    # Test for tan with two Dual objects
    val1 = Dual(3, [4, 1])
    val2 = Dual(2, [3, 1])
    val = val1 - val2
    z = Elem.tan(val)

    try:
        assert z.val == np.tan(val.val)
        assert z.der[0] == 1/np.cos(val.val)**2 * val.der[0]
        assert z.der[1] == 1/np.cos(val.val)**2 * val.der[1]

    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for tan with int,
    x = 3
    fx = Elem.tan(x)
    try:
        assert fx == np.tan(x)

    except AssertionError as e:
        print(e)
        raise AssertionError


def test_log():
    """Test of log method."""
    # Test for log with two Dual objects
    val1 = Dual(3, [4, 1])
    val2 = Dual(2, [3, 1])
    val = val1 * val2
    z = Elem.log(val)

    try:
        assert z.val == np.log(val.val)
        assert z.der[0] == 1/val.val * val.der[0]
        assert z.der[1] == 1/val.val * val.der[1]

    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for log with invalid Dual objects
    with pytest.raises(ValueError, match=r".* logarithm .*"):
        Elem.log(Dual(-1, [4, 1]))

    # Test for log with int,
    x = 3
    fx = Elem.log(x)
    try:
        assert fx == np.log(x)

    except AssertionError as e:
        print(e)
        raise AssertionError


def test_log10():
    """Test of log10 method."""
    # Test for log10 with two Dual objects
    val1 = Dual(3, [4, 1])
    val2 = Dual(2, [3, 1])
    val = val1 * val2
    z = Elem.log10(val)

    try:
        assert z.val == np.log10(val.val)
        assert z.der[0] == 1/(val.val*np.log(10)) * val.der[0]
        assert z.der[1] == 1/(val.val*np.log(10)) * val.der[1]

    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for log10 with invalid Dual objects
    with pytest.raises(ValueError, match=r".* logarithm .*"):
        Elem.log10(Dual(-1, [4, 1]))

    # Test for log10 with int,
    x = 3
    fx = Elem.log10(x)
    try:
        assert fx == np.log10(x)

    except AssertionError as e:
        print(e)
        raise AssertionError


def test_log2():
    """Test of log2 method."""
    # Test for log2 with two Dual objects
    val1 = Dual(3, [4, 1])
    val2 = Dual(2, [3, 1])
    val = val1 * val2
    z = Elem.log2(val)

    try:
        assert z.val == np.log2(val.val)
        assert z.der[0] == 1/(val.val*np.log(2)) * val.der[0]
        assert z.der[1] == 1/(val.val*np.log(2)) * val.der[1]

    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for log2 with invalid Dual objects
    with pytest.raises(ValueError, match=r".* logarithm .*"):
        Elem.log2(Dual(-1, [4, 1]))

    # Test for log2 with int,
    x = 3
    fx = Elem.log2(x)
    try:
        assert fx == np.log2(x)

    except AssertionError as e:
        print(e)
        raise AssertionError


def test_sinh():
    """Test of sinh method."""
    # Test for sinh with two Dual objects
    val = Dual(3, [4, 1])
    z = Elem.sinh(val)

    try:
        assert z.val == np.sinh(val.val)
        assert z.der[0] == np.cosh(val.val) * val.der[0]
        assert z.der[1] == np.cosh(val.val) * val.der[1]

    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for sinh with int
    x = 3
    fx = Elem.sinh(x)
    try:
        assert fx == np.sinh(x)

    except AssertionError as e:
        print(e)
        raise AssertionError


def test_cosh():
    """Test of cosh method."""
    # Test for cosh with two Dual objects
    val = Dual(3, [4, 1])
    z = Elem.cosh(val)

    try:
        assert z.val == np.cosh(val.val)
        assert z.der[0] == np.sinh(val.val) * val.der[0]
        assert z.der[1] == np.sinh(val.val) * val.der[1]

    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for cosh with int,
    x = 3
    fx = Elem.cosh(x)
    try:
        assert fx == np.cosh(x)

    except AssertionError as e:
        print(e)
        raise AssertionError


def test_tanh():
    """Test of tanh method."""
    # Test for tan with two Dual objects
    val = Dual(3, [4, 1])
    z = Elem.tanh(val)
    der = val.der/np.cosh(val.val)

    try:
        assert z.val == np.tanh(val.val)
        assert np.all(z.der == der)

    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for tanh with int,
    x = 3
    fx = Elem.tanh(x)
    try:
        assert fx == np.tanh(x)

    except AssertionError as e:
        print(e)
        raise AssertionError


def test_logistic():
    """Test of logistic method."""
    # Test for logistic with two Dual objects
    x = Dual(3, [4, 1])
    z = Elem.logistic(x)
    result = (1 / (1 + np.exp(-x.val)), np.exp(x.val) / ((1 + np.exp(x.val)) ** 2))
    try:
        assert z == result

    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for logistic with int
    x = 3
    fx = Elem.logistic(x)
    try:
        assert fx == 1 / (1 + np.exp(-x))

    except AssertionError as e:
        print(e)
        raise AssertionError


def test_exp():
    """Test of exp method."""
    # Test for exp with two Dual objects
    x = Dual(3, [4, 1])
    z = Elem.exp(x)
    der = np.exp(x.val) * x.der

    try:
        assert z.val == np.exp(x.val)
        assert np.all(z.der == der)

    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for exp with int,
    x = 3
    fx = Elem.exp(x)
    try:
        assert fx == np.exp(x)

    except AssertionError as e:
        print(e)
        raise AssertionError


# def test_sqrt():
#     """Test of sqrt method."""
#     # Test for sqrt with two Dual objects
#     x = Dual(3, [4, 1])
#     z = Elem.sqrt(x)
#     result = x ** 0.5
#     print(z.der[0])
#     print(z.der[1] )
#     # try:
#     #     assert z.val == np.sqrt(x.val)
#     #     assert z.der[0] == result.der[0]
#     #     assert z.der[1] == result.der[1]
#     #
#     # except AssertionError as e:
#     #     print(e)
#     #     raise AssertionError
#
#     # Test for logistic with int
#     x = 3
#     fx = Elem.sqrt(x)
#     try:
#         assert fx == np.sqrt(x)
#
#     except AssertionError as e:
#         print(e)
#         raise AssertionError


def test_arcsin():
    """Test of arcsin method."""
    # Test for arcsin with two Dual objects
    # arsin() input (-1,1)
    x = Dual(0.2, [0.4, 0.1])
    z = Elem.arcsin(x)
    print(z)
    der = 1 / np.sqrt(1 - x.val ** 2) * x.der
    try:
        assert z.val == np.arcsin(x.val)
        assert np.all(z.der == der)

    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for arcsin with int
    x = 0.1
    fx = Elem.arcsin(x)
    try:
        assert fx == np.arcsin(x)

    except AssertionError as e:
        print(e)
        raise AssertionError


def test_arccos():
    """Test of arccos method."""
    # Test for arccos with two Dual objects
    # arccos() input (-1,1)
    x = Dual(0.2, [0.4, 0.1])
    z = Elem.arccos(x)
    print(z)
    der = -1 / np.sqrt(1 - x.val**2) * x.der
    try:
        assert z.val == np.arccos(x.val)
        assert np.all(z.der == der)

    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for arccos with int
    x = 0.1
    fx = Elem.arccos(x)
    try:
        assert fx == np.arccos(x)

    except AssertionError as e:
        print(e)
        raise AssertionError


def test_arctan():
    """Test of arctan method."""
    # Test for arctan with two Dual objects
    # arctan() input (-1,1)
    x = Dual(0.2, [0.4, 0.1])
    z = Elem.arctan(x)
    der = 1 / (1 + x.val**2) * x.der
    try:
        assert z.val == np.arctan(x.val)
        assert np.all(z.der == der)

    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for arctan with int
    x = 0.1
    fx = Elem.arctan(x)
    try:
        assert fx == np.arctan(x)

    except AssertionError as e:
        print(e)
        raise AssertionError
