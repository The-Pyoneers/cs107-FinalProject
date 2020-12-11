"""This is the driver for Farad.

This file contains class AutoDiff and RAutoDiff.

AutoDiff is for forward mode， and it contains two methods：
values() is for getting the value of the function.
forward() is for getting the derivative of the variables via forward AD mode.

RAutoDiff is for reverse mode， and it contains three methods：
values() is for getting the value of the function.
reverse() is for getting the derivative of the variables via reverse AD mode.
forwardpass() is to constructor the tree structure required to perform reverse AD
calculation. forwardpass needs to be called before using values() and reverse().

"""


from farad.dual import Dual
from numbers import Number
from inspect import signature
from farad.rnode import Rnode
import numpy as np


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
        self.vals = []
        self.ders = []
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
        if self.dimensions == 1:  # (1 input -> scalar objective function)

            if self.length == 1:  # (1 input parameter -> univariate)

                try:  # EAFP principle
                    try:  # default assumption is list input
                        for v in val:
                            a = Dual(v, 1)
                            self.vals.append(self.function(a)._val)
                        return self.vals
                    except TypeError:  # defers to float/integer input
                        a = Dual(val, 1)
                        self.vals.append(self.function(a)._val)
                except TypeError:
                    raise TypeError('Only list, float, and integer inputs supported.')

            else:  # (>=2 input parameters -> multivariate)

                if any(isinstance(value, list) for value in val):  # parse and calculate for each list
                    values = []
                    for value in val:
                        values.append(self.values(value))
                    return values
                try:
                    self.vals.append(self.function(*val))
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
                    self.vals.append(values)

            else:

                self.vals = []  # reset values to prevent duplicates
                for i in range(self.dimensions):
                    def inner_func(*args):
                        return self.function(*args)[i]
                    func = AutoDiff(inner_func, dim=1)
                    func.length = self.length
                    self.vals.append(func.values(val))

            return self.vals


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
        self.ders = []   # reset values to prevent duplicates

        if self.dimensions == 1:  # (1 input -> scalar objective function)

            if self.length > 1:  # (1 input parameter -> univariate)

                if any(isinstance(value, list) for value in val):  # parse and calculate for each list
                    values = []
                    for value in val:
                        values.append(self.forward(value))
                    self.ders = values
                    return self.ders
                try:
                    for i in range(self.length):
                        val2 = val.copy()
                        val2[i] = Dual(val2[i], 1)
                        v = self.function(*val2)
                        if type(v) is Dual:
                            self.ders.append(self.function(*val2)._der)
                        else:
                            self.ders.append(0)
                    return self.ders
                except:
                    raise Exception(f'Mismatch between function parameter length: {self.length}, and input length: {len(val)}.')

            try:  # EAFP principle
                try:  # default assumption is list input
                    for v in val:
                        a = Dual(v, 1)
                        self.ders.append(self.function(a)._der)
                    return self.ders
                except TypeError:  # defers to float/integer input
                    a = Dual(val, 1)
                    self.ders.append(self.function(a)._der)
                    return self.ders
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
                    self.ders.append(values)

            else:  # (>=2 input parameters -> multivariate)

                for i in range(self.dimensions):
                    def inner_func(*args):
                        return self.function(*args)[i]
                    func = AutoDiff(inner_func, dim=1)
                    func.length = self.length
                    self.ders.append(func.forward(val))
            return self.ders


class RAutoDiff:
    def __init__(self, fn):
        """Constructor for RAutoDiff class.

        Parameters
        ==========
        fn: The specific AD method for calculating the derivative.
        """
        self.fn = fn
        self._roots = None
        self._value = None
        self._der = None

    def forwardpass(self, x):
        """Constructor the tree structure with input X for specific AD method
        fn. Update the value and derivative of the AD method.

        Parameters
        ==========
        x: array_like

        Returns:
        No returns.

        """
        self._roots = None
        self._value = None
        self._der = None
        x = np.asarray(x)
        try:
            nf = len(self.fn)  # if fn is a list of functions
        except TypeError:
            nf = 1
        if nf == 1:  # only one function
            self._value, self._der = self._forwardpass1f(x, self.fn)
        else:  # vector functions
            #for now, all the input functions must contain exactly the same parameters
            #with the same order. function with only a subset of total Parameters
            # is not allowed
            nparams_all = [len(signature(fi).parameters) for fi in self.fn]
            if len(set(nparams_all)) > 1:
                raise TypeError('all input functions must contain the same parameters')
            self._value = []
            self._der = []
            for idxf in range(nf):
                tmp = self._forwardpass1f(x, self.fn[idxf])
                self._value.append(tmp[0])
                self._der.append(tmp[1])
        self._value = np.asarray(self._value)
        self._der = np.asarray(self._der)
        if len(self._value.shape) == 2 and len(self._der.shape) == 2:
            self._value = np.transpose(self._value, (1, 0))
            self._der = np.transpose(self._der, (1, 0))
        elif len(self._value.shape) == 2 and len(self._der.shape) == 3:
            self._value = np.transpose(self._value, (1, 0))
            self._der = np.transpose(self._der, (1, 0, 2))

        try:  # if _value or _der is a scalar array (size = 1), convert array to scalar
            if self._value.size ==1:
                self._value = np.asscalar(self._value)
            if self._der.size == 1:
                self._der = np.asscalar(self._der)
        except AttributeError:
            pass

    def _forwardpass1f(self, x, fi):  # deal with only one function case
        """Constructor the tree structure with input X for one single AD method
        fi. This _forwardpass1f method will be called when dealing with vector function
        input fn = [f1, f2, f3, ...]

        Parameters
        ==========
        x: array_like
        fi: One AD method

        Returns:
        tmpval: array_like, the value of fi(x)
        tmpder : array_like, the derivative of fi(x)

        """
# shoule only be called from forwardpass
        self._roots = None
        nparams = len(signature(fi).parameters)  # number of parameters of input function
        if nparams == 1:  # function has only one parameter
            if x.size == 1:  # scalar input
                self._roots = Rnode(x)
                f = fi(self._roots)
                f.grad_value = 1.0
                tmpval = f.value
                tmpder = self._roots.grad()
            else:  # vector input
                if len(x.shape) > 1:
                    raise TypeError('input dimension size not supported')
                else:
                    tmpval = np.zeros(len(x))
                    tmpder = np.zeros(len(x))
                    tmpidx = 0
                    for xi in x:
                        self._roots = Rnode(xi)
                        f = fi(self._roots)
                        f.grad_value = 1.0
                        tmpval[tmpidx] = f.value
                        tmpder[tmpidx] = self._roots.grad()
                        tmpidx = tmpidx + 1

        else:  # multiple input parameters (vector input)
            if x.size == 1:
                raise TypeError('input has insufficient parameters')
            if len(x.shape) == 1:  # evaluate at one vector point
                if len(x) != nparams:
                    raise TypeError('input dimension size mismatch')
                self._roots = [Rnode(xi) for xi in x]
                f = fi(*self._roots)
                f.grad_value = 1.0
                tmpval = f.value
                tmpder = [root.grad() for root in self._roots]
            else:  # evaluate at multiple vector points
                if x.shape[1] != nparams:
                    raise TypeError('input dimension size mismatch')
                nm = x.shape[0]  # number of vector points to be evaluated
                tmpval = np.zeros(nm)
                tmpder = np.zeros((nm, nparams))
                self._roots = []
                for im in range(nm):
                    self._roots.append([Rnode(xi) for xi in x[im, :]])
                    f = fi(*self._roots[im])
                    f.grad_value = 1.0
                    tmpval[im] = f.value
                    tmpder[im, :] = [root.grad() for root in self._roots[im]]
        return tmpval, tmpder

    def values(self):  # return the value of the function
        """Get value of the input method fn for given X

        Returns:
        y : array_like

        """
        return self._value

    def reverse(self):  # return the derivative with respect to varname variable
        """Get the derivatives (scalar, vector, or matrix depending on the
        the dimension of input method fn and X)

        Returns:
        y : array_like

        """
        return self._der
