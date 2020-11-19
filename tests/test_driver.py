"""Test Math functionality for farad package.
"""

import pytest
import numpy as np
import farad.elem as Elem
from farad.dual import Dual
import farad.driver as ad

def test_get_val_scalar():
    """Test of get_val_scalar method."""
    def f(x):
        return exp(x)
    function = ad.AutoDiff(f)

    # test scalar input
    try:
        assert function.get_val_scalar(1) == np.exp(1)
    except AssertionError as e:
        print(e)
        raise AssertionError

    # test list input
    try:
        assert function.get_val_scalar([1,2,3]) == np.exp([1,2,3])
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_forward():
    """Test of forward method."""
    def f(x):
        return exp(x)
    function = ad.AutoDiff(f)

    # test scalar input
    try:
        assert function.forward(1) == np.exp(1)
    except AssertionError as e:
        print(e)
        raise AssertionError

    # test list input
    try:
        assert function.forward([1,2,3]) == np.exp([1,2,3])
    except AssertionError as e:
        print(e)
        raise AssertionError
