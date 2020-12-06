"""This is the driver of reverse mode for Project Farad.

This file contains class RAutoDiff.
get_var() is for getting value of a function.
reverse() is for getting the derivative of the variables via reverse AD mode.

Currently multiple situations are supported，with the function type and dimension size of input and output are listed below:

function form, x, function value, function derivatives
f(x)           1               1                     1
f(x)           m               m                     m
f(xn)          n               1                     n
f(xn)        mxn               m                   mxn
fk(x)          1               k                     k
fk(x)          m             mxk                   mxk
fk(xn)         n               k                   kxn
fk(xn)       mxn             mxk                 mxkxn

m is when x (either scalar or vector) is evaluated at m points
n is when x is a vector with length n
k is when input function is a vector function with length k
"""


from farad.elem import *
from numbers import Number
from inspect import signature
import numpy as np
from farad.rnode import Rnode


class RAutoDiff:
    def __init__(self, fn):
        self.fn = fn
        self._roots = None
        self._value = None
        self._der = None

    def forwardpass(self, x):
        self._roots = None
        self._value = None
        self._der = None
        if not isinstance(x, Number):
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
# shoule only be called from forwardpass
        self._roots = None
        nparams = len(signature(fi).parameters)  # number of parameters of input function
        if nparams == 1:  # function has only one parameter
            if isinstance(x, Number):
                self._roots = Rnode(x)
                f = fi(self._roots)
                f.grad_value = 1.0
                tmpval = f.value
                tmpder = self._roots.grad()
            else:
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
            if isinstance(x, Number):
                raise TypeError('input dimension size mismatch')
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
        return self._value

    def reverse(self):  # return the derivative with respect to varname variable
        return self._der

if __name__ == "__main__":
    #test for driver

    def fi(x):
        return x+sin(x)*exp(cos(x))

    autofx = RAutoDiff(fi)
    autofx.forwardpass(4)
    print('value of fi is:', autofx.get_val())
    print('value of fi derivative is:', autofx.reverse())

    # evaluate fi(x) at different x values
    autofx.forwardpass([1,2,3,4])
    print('value of fi is:', autofx.get_val())
    print('value of fi derivatives is:', autofx.reverse())


    #evaluate f(x,y) at (x,y)
    def fi2d(x, y):
        return x+sin(x) + x * y

    autofx2d = RAutoDiff(fi2d)
    autofx2d.forwardpass([1,2])
    print('value of fi2 is:', autofx2d.get_val())
    print('value of fi2d derivatives are:', autofx2d.reverse())

    autofx2d = RAutoDiff(fi2d)
    autofx2d.forwardpass([[1,2],[3,4],[5,6]])
    print('value of fi2 is:', autofx2d.get_val())
    print('value of fi2d derivatives are:', autofx2d.reverse())

    # evaluate f = [f1(x,y), f2(x)] at (x,y)
    #def fi2d(x, y):
    #    return x+sin(x) + x * y
    print('for vector function:')
    fvector = [fi2d, fi2d]
    autofxv = RAutoDiff(fvector)
    autofxv.forwardpass([[1,2],[1,2],[1,2]])
    print('value of fvector is: \n', autofxv.get_val())
    print('value of fvector derivatives are: \n', autofxv.reverse())

    #autofx2d = RAutoDiff(fi2d)
    #autofx2d.forwardpass([1,2])
    #print('value of fi is:', autofx2d.get_val())
    #print('value of derivatives are:', autofx2d.reverse())
