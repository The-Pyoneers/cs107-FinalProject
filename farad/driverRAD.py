"""This is the driver of reverse mode for Project Farad.

This file contains class RAutoDiff.
get_var() is for getting value of a function.
reverse() is for getting the derivative of the variables via reverse AD mode.

Currently multiple situations are supported，with the function type and dimension size of input and output are listed below:

function form F, X, function value, function derivatives
F(X)           1               1                     1
F(X)           m               m                     m
F(Xn)          n               1                     n
F(Xn)        mxn               m                   mxn
Fk(X)          1               k                     k
Fk(X)          m             mxk                   mxk
Fk(Xn)         n               k                   kxn
Fk(Xn)       mxn             mxk                 mxkxn

m: X (either scalar or vector) is evaluated at m points
n: X is a vector with length n
k: input function F is a vector function with length k
"""


from farad.elem import *
from numbers import Number
from inspect import signature
import numpy as np
from farad.rnode import Rnode


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
        else: # vector functions
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
            if x.size == 1: # scalar input
                self._roots = Rnode(x)
                f = fi(self._roots)
                f.grad_value = 1.0
                tmpval = f.value
                tmpder = self._roots.grad()
            else: # vector input
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

    def get_val(self):  # return the value of the function
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
