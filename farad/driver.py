"""This is the driver for Project Farad.

This file contains class AutoDiff which has two methods.
get_var_scalar() is for getting scalar variable including a number or a list.
forward() is for getting the derivative of the variables via forward AD mode.

Note that for milestone2 this class only deals with scalar, the function will be extended in next stage.
"""


from farad.dual import Dual
from numbers import Number
from farad.elem import *


class AutoDiff:

    def __init__(self, fn):
        """Constructor for AutoDiff class.

        Parameters
        ==========
        fn: The specific AD method for calculating the derivative

        Notes
        =====
        Currently, only first order derivatives of scalar functions via
        forward mode are supported. The methods and the range of functions
        will be extended in later versions to handle vector inputs with Jacobians.
        """
        self.fn = fn

    def get_val_scalar(self, val):
        """Get a scalar or a list of scalars.

        Args:
            val: a scalar or a list of scalars.

        Returns: self.fn(a).val or list of self.fn(a).val.

        """
        # evaluate a number at a single variable.
        if isinstance(val, Number):
            a = Dual(val, 1)
            return self.fn(a).val
        # evaluate a list of numbers at a single variable.
        elif isinstance(val, list):
            vals = []
            for v in val:
                a = Dual(v, 1)
                vals.append(self.fn(a).val)
            return vals
        else:
            raise Exception("Currently not support types other than number and list")

    def forward(self, val):
        """Forward mode method of AutoDiff class.

        Args:
            val: a scalar or a list of scalars.

        Returns: self.fn(a).ders or list of self.fn(a).ders.
        The derivatives of val.
        """
        # Get the derivative of a number at a single variable
        if isinstance(val, Number):
            a = Dual(val, 1)
            return self.fn(a).der
        # Get the derivative of a list of numbers at a single variable
        elif isinstance(val, list):
            ders = []
            for v in val:
                a = Dual(v, 1)
                ders.append(self.fn(a).der)
            return ders
        else:
            raise Exception("Currently not support input types other than number and list")
