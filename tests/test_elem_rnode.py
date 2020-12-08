import pytest
import farad.elem as Elem
import numpy as np
from farad.rnode import Rnode

def test_sin_rnode():
    """Test of sin method for reverse mode."""
    # Test for sin with two Dual objects
    x = Rnode(1.0)
    z = Elem.sin(x)
    z.grad_value = 1.0
    try:
        assert z.value == np.sin(x.value)
        assert x.grad() == np.cos(x.value)

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
