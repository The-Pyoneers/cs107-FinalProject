"""Test Math functionality for farad package.
"""

import pytest
import numpy as np
import farad.elem as Elem
from farad.dual import Dual
from farad.rnode import Rnode


def test_sin():
    """Test of sin method."""
    # Test for sin with Rnode objects

    x = Rnode(1.0)
    z = Elem.sin(x)
    z.grad_value = 1.0
    try:
        assert z.value == np.sin(x.value)
        assert x.grad() == np.cos(x.value)

    except AssertionError as e:
        print(e)
        raise AssertionError

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
    # Test for sin with Rnode objects

    x = Rnode(1.0)
    z = Elem.cos(x)
    z.grad_value = 1.0
    try:
        assert z.value == np.cos(x.value)
        assert x.grad() == -np.sin(x.value)

    except AssertionError as e:
        print(e)
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
    # Test for sin with Rnode objects

    x = Rnode(1.0)
    z = Elem.tan(x)
    z.grad_value = 1.0
    try:
        assert z.value == np.tan(x.value)
        assert x.grad() == 1 / (np.cos(x.value) ** 2)

    except AssertionError as e:
        print(e)
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
    # Test for sin with Rnode objects

    x = Rnode(1.0)
    z = Elem.log(x)
    z.grad_value = 1.0
    try:
        assert z.value == np.log(x.value)
        assert x.grad() == 1/x.value

    except AssertionError as e:
        print(e)
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
    # Test for sin with Rnode objects

    x = Rnode(1.0)
    z = Elem.log10(x)
    z.grad_value = 1.0
    try:
        assert z.value == np.log10(x.value)
        assert x.grad() == 1/(x.value * np.log(10))

    except AssertionError as e:
        print(e)
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
    # Test for sin with Rnode objects

    x = Rnode(1.0)
    z = Elem.log2(x)
    z.grad_value = 1.0
    try:
        assert z.value == np.log2(x.value)
        assert x.grad() == 1/(x.value * np.log(2))

    except AssertionError as e:
        print(e)
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
    # Test for sin with Rnode objects

    x = Rnode(1.0)
    z = Elem.sinh(x)
    z.grad_value = 1.0
    try:
        assert z.value == np.sinh(x.value)
        assert x.grad() == np.cosh(x.value)

    except AssertionError as e:
        print(e)
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
    # Test for sin with Rnode objects

    x = Rnode(1.0)
    z = Elem.cosh(x)
    z.grad_value = 1.0
    try:
        assert z.value == np.cosh(x.value)
        assert x.grad() == np.sinh(x.value)

    except AssertionError as e:
        print(e)
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
    # Test for sin with Rnode objects

    x = Rnode(1.0)
    z = Elem.tanh(x)
    z.grad_value = 1.0
    try:
        assert z.value == np.tanh(x.value)
        assert x.grad() == 1 / np.cosh(x.value)**2

    except AssertionError as e:
        print(e)
    # Test for tan with two Dual objects
    val = Dual(3, [4, 1])
    z = Elem.tanh(val)
    der = val.der/(np.cosh(val.val))**2

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


def test_relu():
    """Test of relu method."""
    # Test for sin with Rnode objects

    x = Rnode(1.0)
    z = Elem.relu(x)
    z.grad_value = 1.0

    a = max(0, x.value)
    b = np.where(a > 0, 1, 0)
    try:
        assert z.value == a
        assert x.grad() == b

    except AssertionError as e:
        print(e)
    # Test for relu with two Dual objects
    x = Dual(3, [4, 1])
    z = Elem.relu(x)

    a = max(0, x.val)
    b = np.where(a > 0, 1, 0)
    result = Dual(a, b * x.der)

    try:
        assert z.val == result.val
        assert np.all(z.der == result.der)

    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for tanh with int,
    x = 3
    fx = Elem.relu(x)
    try:
        assert fx == max(0, x)

    except AssertionError as e:
        print(e)
        raise AssertionError


def test_relu6():
    """Test of relu6 method."""
    # Test for sin with Rnode objects

    x = Rnode(7.0)
    z = Elem.relu6(x)
    z.grad_value = 1.0

    a = max(0, x.value)
    b = np.where(0.0 < a < 6.0, 1, 0)
    if a > 6.0:  # clip output to a maximum of 6
        a = 6.0
    try:
        assert z.value == a
        assert x.grad() == b

    except AssertionError as e:
        print(e)
    # Test for relu6 with two Dual objects
    x = Dual(8, [4, 1])
    z = Elem.relu6(x)

    a = max(0, x.val)
    b = np.where(0.0 < a < 6.0, 1, 0)
    print(b)
    if a > 6:  # clip output to a maximum of 6
        a = 6
    result = Dual(a, b * x.der)

    try:
        assert z.val == result.val
        assert np.all(z.der == result.der)

    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for tanh with int,
    x = 3
    fx = Elem.relu6(x)
    try:
        assert fx == min(max(0, x), 6)

    except AssertionError as e:
        print(e)
        raise AssertionError


def test_logistic():
    """Test of logistic method."""
    # Test for sin with Rnode objects

    x = Rnode(1.0)
    z = Elem.logistic(x)
    z.grad_value = 1.0

    nominator = np.exp(x.value)
    denominator = (1 + np.exp(x.value)) ** 2
    try:
        assert z.value == 1 / (1 + np.exp(-x.value))
        assert x.grad() == nominator / denominator

    except AssertionError as e:
        print(e)
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
    # Test for sin with Rnode objects

    x = Rnode(1.0)
    z = Elem.exp(x)
    z.grad_value = 1.0

    try:
        assert z.value == np.exp(x.value)
        assert x.grad() == np.exp(x.value)

    except AssertionError as e:
        print(e)
    # Test for exp with two Dual objects
    x = Dual(3, [4, 1])
    z = Elem.exp(x)
    der = np.exp(x.val) * np.asarray(x.der)

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


# def test_exp2():
#     """Test of exp2 method."""
#     # Test for exp2 with two Dual objects
#     x = Dual(3, [4, 1])
#     z = Elem.exp2(x)
#     print(z)
#     der = np.exp2(x._val) * (x._der * np.log(2))
#
#     try:
#         assert z.val == np.exp2(x.val)
#         assert np.all(z.der == der)
#
#     except AssertionError as e:
#         print(e)
#         raise AssertionError
#
#     # Test for exp2 with int,
#     x = 3
#     fx = Elem.exp2(x)
#     try:
#         assert fx == np.exp2(x)
#
#     except AssertionError as e:
#         print(e)
#         raise AssertionError


def test_sqrt():
    """Test of sqrt method."""
    # Test for sqrt with Rnode objects

    x = Rnode(1.0)
    z = Elem.sqrt(x)
    z.grad_value = 1.0

    try:
        assert z.value == x.value ** 0.5
        assert x.grad() == 0.5*x.value ** (-0.5)

    except AssertionError as e:
        print(e)
    # Test for sqrt with two Dual objects
    x = Dual(3, [4, 1])
    z = Elem.sqrt(x)
    result = x ** 0.5

    try:
        assert z.val == np.sqrt(x.val)
        assert z.der[0] == result.der[0]
        assert z.der[1] == result.der[1]

    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for sqrt with double
    x = 3.0
    fx = Elem.sqrt(x)
    try:
        assert fx == np.sqrt(x)

    except AssertionError as e:
        print(e)
        raise AssertionError


def test_power():
    """Test of power method."""
    # Test for power with two Dual objects
    x = Dual(3, [4, 1])
    z = Elem.power(x, 2)
    result = x ** 2

    try:
        assert z.val == np.power(x.val, 2)
        assert z.der[0] == result.der[0]
        assert z.der[1] == result.der[1]

    except AssertionError as e:
        print(e)
        raise AssertionError

    # Test for power with double
    x = 3.0
    fx = Elem.power(x, 2.0)
    try:
        assert fx == 9.0

    except AssertionError as e:
        print(e)
        raise AssertionError


def test_arcsin():
    """Test of arcsin method."""
    # Test for arcsin with Rnode objects

    x = Rnode(0.11)
    z = Elem.arcsin(x)
    z.grad_value = 1.0
    temp = 1 - x.value ** 2
    if temp <= 0:
        raise ValueError('Domain of sqrt is {x >= 0}')
    try:
        assert z.value == np.arcsin(x.value)
        assert x.grad() == 1 / np.sqrt(temp)
    except AssertionError as e:
        print(e)

    # Test for arcsin with invalid Rnode objects
    with pytest.raises(ValueError, match=r".* sqrt .*"):
        Elem.arcsin(Rnode(1.0))


    # Test for arcsin with two Dual objects
    # arsin() input (-1,1)
    x = Dual(0.2, [0.4, 0.1])
    z = Elem.arcsin(x)
    print(z)
    der = 1 / np.sqrt(1 - x.val ** 2) * np.asarray(x.der)
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
    # Test for arccos with Rnode objects

    x = Rnode(0.11)
    z = Elem.arccos(x)
    z.grad_value = 1.0
    temp = 1 - x.value ** 2
    print(temp)
    if temp <= 0:
        raise ValueError('Domain of sqrt is {x >= 0}')
    try:
        assert z.value == np.arccos(x.value)
        assert x.grad() == -1 / np.sqrt(temp)
    except AssertionError as e:
        print(e)

    # Test for arccos with invalid Rnode objects
    with pytest.raises(ValueError, match=r".* sqrt .*"):
        x = Rnode(2.0)
        z = Elem.arccos(x)
        Elem.arcsin(x)
        z.grad_value = 1.0

    # Test for arccos with two Dual objects
    # arccos() input (-1,1)
    x = Dual(0.2, [0.4, 0.1])
    z = Elem.arccos(x)
    print(z)
    der = -1 / np.sqrt(1 - x.val**2) * np.asarray(x.der)
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
    # Test for arctan with Rnode objects

    x = Rnode(0.11)
    z = Elem.arctan(x)
    z.grad_value = 1.0

    try:
        assert z.value == np.arctan(x.value)
        assert x.grad() == 1 / (1 + x.value ** 2)
    except AssertionError as e:
        print(e)

    # Test for arctan with two Dual objects
    # arctan() input (-1,1)
    x = Dual(0.2, [0.4, 0.1])
    z = Elem.arctan(x)
    der = 1 / (1 + x.val**2) * np.asarray(x.der)
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
