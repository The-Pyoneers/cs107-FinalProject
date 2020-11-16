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
