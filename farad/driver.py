"""This is the driver for Project Farad.

This file contains class AutoDiff which has two methods.
get_var_scalar() is for getting scalar variable including a number or a list.
forward() is for getting the derivative of the variables via forward AD mode.

Note that for milestone2 this class only deals with scalar, the function will be extended in next stage.
"""


from farad.dual import Dual
from numbers import Number


class AutoDiff:

    def __init__(self, fn):
        self.fn = fn

    def get_val_scalar(self, val):
        # for a number, evaluated at a single variable.
        if isinstance(val, Number):
            a = Dual(val, 1)
            return self.fn(a).val
        # for a list of numbers, evaluated at a single variable.
        elif isinstance(val, list):
            vals = []
            for v in val:
                a = Dual(v, 1)
                vals.append(self.fn(a).val)
            return vals
        else:
            raise Exception("Currently we have not accepted input types other than number and list")

    def forward(self, val):
        # for a number, get the derivative at a single variable
        if isinstance(val, Number):
            a = Dual(val, 1)
            return self.fn(a).der
        # for a list of numbers, get the derivative at a single variable
        elif isinstance(val, list):
            ders = []
            for v in val:
                a = Dual(v, 1)
                ders.append(self.fn(a).der)
                return ders
        else:
            raise Exception("Currently we have not accepted input types other than number and list")
