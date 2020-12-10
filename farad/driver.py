"""This is the driver for Farad.

This file contains class AutoDiff which has two methods.
get_var_scalar() is for getting scalar variable including a number or a list.
forward() is for getting the derivative of the variables via forward AD mode.

Note that for milestone2 this class only deals with scalar, the function will be extended in next stage.
"""


from farad.dual import Dual
from numbers import Number
from inspect import signature
from farad.elem import *


class AutoDiff(object):


    def __init__(self, function, dim=1):
        """Constructor for AutoDiff class.

        Parameters
        ==========
        function: The specific AD method for calculating the derivative.
        dim: The dimensionality of the function.
        
        Notes
        =====
        Currently, only first order derivatives of scalar functions via
        forward mode are supported. The methods and the range of functions
        will be extended in later versions to handle vector inputs with Jacobians.
        """
        self.function = function  # function input (e.g., lambda x, y: x**2 + y**2)
        self.dimensions = dim  # dimensionality of function input (e.g., 2 for lambda x: [x**2, x**3])
        self._vals = []
        self._ders = []
        try:
            # if defined as lambda function
            self.length = len(self.function.__code__.co_varnames)  # no. of function inputs (e.g., 2 for lambda x, y: x**2 +  y**2)
        except AttributeError:  # if defined as standard function
            self.length = signature(self.function)


    def values(self, val):
        """Returns node values after propagation through the computational graph.

        Parameters
        ==========
        val: a float/integer scalar or a list of scalars.
            Values to be propagated from the input node through the computational graph.

        Returns
        =======
        list[float|int], float, or int.
            Returns list of float or integer values corresponding to output node values.

        Examples
        ========
        >>> example = AutoDiff(lambda x: 2**x)
        >>> example.values(3)
        8
        >>> example = AutoDiff(lambda x: 2**x)
        >>> example.values([3, 4])
        [8, 16]
        >>> example = AutoDiff(lambda x, y: 3*x + 2*y)
        >>> example.values([3, 4])
        17
        >>> example = AutoDiff(lambda x, y: 2*x + 4*y)
        >>> example.values([[3, 4], [5,4]])
        [22, 26]
        """
        #self._vals = []  # reset values to prevent duplicates

        if self.dimensions == 1:  # (1 input -> scalar objective function)

            if self.length == 1:  # (1 input parameter -> univariate)
                
                try:  # EAFP principle
                    try:  # default assumption is list input
                        for v in val:
                            a = Dual(v, 1)
                            self._vals.append(self.function(a)._val)
                        return self._vals
                    except TypeError:  # defers to float/integer input
                        a = Dual(val, 1)
                        return self.function(a)._val
                except TypeError:
                    raise TypeError('Only list, float, and integer inputs supported.')

            else:  # (>=2 input parameters -> multivariate)

                if any(isinstance(value, list) for value in val):  # parse and calculate for each list
                    values = []
                    for value in val:
                        values.append(self.values(value))
                    return values
                try:
                    return self.function(*val)
                except:
                    raise Exception(f'Mismatch between function parameter length: {self.length}, and input length: {len(val)}.')

        else:  # (>=2 inputs -> vector objective function)

            if any(isinstance(el, list) for el in val):
                for v in val:
                    values = []
                    for i in range(self.dimensions):
                        def inner_func(*args):
                            return self.function(*args)[i]
                        func = AutoDiff(inner_func, dim=1)
                        func.length = self.length
                        values.append(func.values(v))
                    self._vals.append(values)

            else:
                
                self._vals = []  # reset values to prevent duplicates
                for i in range(self.dimensions):
                    def inner_func(*args):
                        return self.function(*args)[i]
                    func = AutoDiff(inner_func, dim=1)
                    func.length = self.length
                    self._vals.append(func.values(val))
            return self._vals


    def forward(self, val):
        """Forward mode method of AutoDiff class.

        Parameters
        ==========
        val: a float/integer scalar or a list of scalars.
            Derivatives to be propagated from the input node through the computational graph.

        Returns
        =======
        list[float|int], float, or int.
            Returns list of float or integer values corresponding to output node derivatives.

        Examples
        ========
        >>> example = AutoDiff(lambda x: 2**x)
        >>> example.forward(3)
        5.545177444479562
        >>> example = AutoDiff(lambda x: 2**x)
        >>> example.forward([3, 4])
        [5.545177444479562, 11.090354888959125]
        >>> example = AutoDiff(lambda x, y: 3*x + 2*y)
        >>> example.forward([3, 4])
        [3, 2]
        >>> example = AutoDiff(lambda x, y: 2*x + 4*y)
        >>> example.forward([[3, 4], [5,4]])
        [[2, 4], [2, 4]]
        """
        self._ders = []   # reset values to prevent duplicates

        if self.dimensions == 1:  # (1 input -> scalar objective function)

            if self.length > 1:  # (1 input parameter -> univariate)

                if any(isinstance(value, list) for value in val):  # parse and calculate for each list
                    values = []
                    for value in val:
                        values.append(self.forward(value))
                    self._ders = values
                    return self._ders
                try:
                    for i in range(self.length):
                        val2 = val.copy()
                        val2[i] = Dual(val2[i], 1)
                        v = self.function(*val2)
                        if type(v) is Dual:
                            self._ders.append(self.function(*val2)._der)
                        else:
                            self._ders.append(0)
                    return self._ders
                except:
                    raise Exception(f'Mismatch between function parameter length: {self.length}, and input length: {len(val)}.')
        
            try:  # EAFP principle
                try:  # default assumption is list input
                    for v in val:
                        a = Dual(v, 1)
                        self._ders.append(self.function(a)._der)
                    return self._ders
                except TypeError:  # defers to float/integer input
                    a = Dual(val, 1)
                    return self.function(a)._der
            except TypeError:
                raise TypeError('Only list, float, and integer inputs supported.')

        else:  # (>=2 inputs -> vector objective function)

            if any(isinstance(el, list) for el in val):
                for v in val:
                    values = []
                    for i in range(self.dimensions):
                        def inner_func(*args):
                            return self.function(*args)[i]
                        func = AutoDiff(inner_func, dim=1)
                        func.length = self.length
                        values.append(func.forward(v))
                    self._ders.append(values)

            else:  # (>=2 input parameters -> multivariate)
                
                for i in range(self.dimensions):
                    def inner_func(*args):
                        return self.function(*args)[i]
                    func = AutoDiff(inner_func, dim=1)
                    func.length = self.length
                    self._vals.append(func.forward(val))
            return self._ders
