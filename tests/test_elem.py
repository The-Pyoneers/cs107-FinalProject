import pytest
import farad.elem as Elem
from farad.dual import Dual

import numpy as np

# val1 = Dual(3,[4,1])
# val = val1 + val2
# val2 = Dual(2,[3,1])
# print(val)
# z = sin(val)
# print(z)
# z = cos(val)
# print(z)
# z = tan(val)
# print(z)
# print(bool(z))


def test_sin():
    """Test of sin method."""
    # Test for sin with two Dual objects
    val1 = Dual(3, [4, 1])
    val2 = Dual(2, [3, 1])
    val = val1 + val2
    print(val)
    z = Elem.sin(val)
    print(z)
    print(np.cos(val.val) * val.der[0])

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
    print(val)
    z = Elem.cos(val)
    print(z)
    print(-np.sin(val.val) * val.der[0])

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
    print(val)
    z = Elem.tan(val)
    print(z)
    print(1/np.cos(val.val)**2 * val.der[0])

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
    print(val)
    z = Elem.log(val)
    print(z)
    print(1/val.val * val.der[0])

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
    print(val)
    z = Elem.log10(val)
    print(z)
    print(1/(val.val*np.log(10)) * val.der[0])

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
    print(val)
    z = Elem.log2(val)
    print(z)
    print(1/(val.val*np.log(2)) * val.der[0])

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
    print(np.cosh(val.val) * val.der[0])

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
