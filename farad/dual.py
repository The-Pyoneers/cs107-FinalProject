#-*- coding: utf-8 -*-
"""Dual number implementation for Python forward AD mode.
​
Author: Matthew Stewart
Date Created: October 30th, 2020
Date Last Updated: October 30th, 2020
​
This module contains dunder methods to overload built-in Python operators
such as addition and multiplication. This system will only be used for the
forward AD mode, which will be called by the FaradObject.forward() method. If
the method fails to implement dual numbers, it will fall back to using standard
methods.
​
Example (exp(x^2) + 2x):
​
        $ import farad as ad
        # import farad.dual as dual
        $ farad_object = dual.exp(dual.var('x')**2) + 2*dual.var('x')  # derivative is 2x*exp(x^2) + 2
        $ farad_object.forward({'x'=0.5},dual=True)  # forward function calls Dual class
        >>> output
​
Attributes:
    val T<numbers.Integral>: Function value evaluated at a fixed point.
    der T<list|np.array>: List of first derivatives evaluated at a fixed point.
​
Todo:
    * For module TODOs
    * Add Sphinx documentation
"""

import numpy as np
import numbers
import reprlib

from typing import NoReturn, List, Union
Array = Union[List[float], np.ndarray]

class Dual():


    def __init__(self, val: numbers.Integral, der: Array) -> NoReturn:
        """Initialize value and derivatives when called."""
        self._val = val
        self._der = np.array(der)


    @property
    def der(self) -> Array:
        return self._der


    @property
    def val(self) -> numbers.Integral:
        return self._val


    def __add__(self, x: "Dual") -> "Dual":
        """Overload the addition operator (+) to handle Dual class."""
        try:  # try-except loop for Python principle EAFP
            return Dual(self._val + x._val, self._der + x._der)
        except AttributeError:
            raise AttributeError(f"Input not a Dual class.")


    def __radd__(self, x: "Dual") -> "Dual":
        """Revert to __add__ dunder method to handle input reversal."""
        return self.__add__(x)


    def __sub__(self, x: "Dual") -> "Dual":
        """Overload the subtraction operator (-) to handle Dual class."""
        try:
            return Dual(self._val - x._val, self._der - x._der)
        except AttributeError:
            raise AttributeError(f"Input not a Dual class.")


    def __rsub__(self, x: "Dual") -> "Dual":
        """Revert to __sub__ dunder method to handle input reversal."""
        return self.__sub__(x)


    def __mul__(self, x: "Dual") -> "Dual":
        """Overload the multiplication operator (*) to handle Dual class."""
        try:
            return Dual(self.val * x.val, self.der * x.val + self.val * x.der)  # chain rule for derivative
        except AttributeError:
            raise AttributeError(f"Input not a Dual class.")


    def __rmul__(self, x: "Dual") -> "Dual":
        """Revert to __mul__ dunder method to handle input reversal."""
        return self.__mul__(x)


    def __pow__(self, x: "Dual") -> "Dual":
        """Overload the exponent operator (**) to handle Dual class."""
        try:
            return Dual(self.val**x.val, self.val**x.val*(self.der*(x.val/self.val) + x.der*np.log(self.val)))
        except AttributeError:
            raise AttributeError(f"Input not a Dual class.")
        except ZeroDivisionError:
            return ZeroDivisionError


    def __rpow__(self, x: "Dual") -> "Dual":
        """Overload input reversed exponent operator to handle Dual class."""
        # Cannot revert to __pow__ dunder method due to non-commutativity of exponent operator
        try:
            return Dual(x**self.val, np.log(x) * x ** self.val  * self.der)
        except AttributeError:
            raise AttributeError(f"Input not a Dual class.")


    def __truediv__(self, x: "Dual") -> "Dual":
        """Overload the division operator (/) to handle Dual class."""
        try:
            return Dual(self.val/ x.val, (self.der * x.val - self.val * x.der)/(x.val)**2)
        except AttributeError:
            return AttributeError(f"Input not a Dual class.")
        except ZeroDivisionError:
            raise ZeroDivisionError


    def __rtruediv__(self, x: "Dual") -> "Dual":
        """Overload input reversed division operator to handle Dual class."""
        # Cannot revert to __truediv__ dunder method due to non-commutativity of divison operator
        try:
            return Dual(x/self.val, -(x * self.der)/self.val ** 2)
        except AttributeError:
            return AttributeError(f"Input not a Dual class.")
        except ZeroDivisionError:
            raise ZeroDivisionError


    def __neg__(self: "Dual") -> "Dual":
        """Overload the unary negation operator (e.g., -x) to handle Dual class."""
        return Dual(-self.val, -self.der)


    def __eq__(self, x: "Dual") -> bool:
        """Overload the equality operator (e.g., x==y) to handle Dual class."""
        try:  # classes equivalent if values and derivatives are equal
            return (self.val == x.val and np.array_equal(self.der, x.der))
        except AttributeError:
            return AttributeError(f"Input not a Dual class.")


    def __neq__(self, x: "Dual") -> bool:
        """Overload the inequality operator (e.g., x!=y) to handle Dual class."""
        try:
            return (self.val != x.val or (np.array_equal(self.der, x.der) == False))
        except AttributeError:
            return AttributeError(f"Input not a Dual class.")


    def __lt__(self, x: "Dual") -> bool:
        """Overload the less than dunder method (e.g., x<y) to handle Dual class."""
        try:
            return (self.val < x.val)
        except AttributeError:
            return (self.val < x)


    def __le__(self, x: "Dual") -> bool:
        """Overload the less than or equal to dunder method (e.g., x<=y) to handle Dual class."""
        try:
            return (self.val <= x.val)
        except AttributeError:
            return (self.val <= x)


    def __gt__(self, x: "Dual") -> bool:
        """Overload the greater than dunder method (e.g., x>y) to handle Dual class."""
        try:
            return (self.val > x.val)
        except AttributeError:
            return (self.val > x)


    def __ge__(self, x: "Dual") -> bool:
        """Overload the greater than or equal to dunder method (e.g., x>=y) to handle Dual class."""
        try:
            return (self.val >= x.val)
        except AttributeError:
            return (self.val >= x)


    def __repr__(self) -> str:
        """Prints class definition with inputs - the output can be passed to eval() to instantiate new instance of class Dual."""
        return f"Dual({reprlib.repr(self.val)},{reprlib.repr(list(self.der))})"


    def __str__(self) -> str:
        """Prints user-friendly class definition."""
        return f"Forward-mode {self.__class__.__name__} Object ( Values: {reprlib.repr(np.round(self.val, 4))}, Derivatives: {reprlib.repr(list(np.round(self.der, 4)))} )"


    def __len__(self) -> str:
        """Returns length of input vector."""
        try:
            return len(self.val)
        except TypeError:
            return 1


    def __bool__(self) -> bool:
        return False


    def __nonzero__(self) -> bool:
        return self.__bool__()


if __name__ == '__main__':
    x = Dual(1, [1,0])
    y = Dual(1, [0,1])
    z = x > y
    #print(z)
    x += y
    #print(x)
    z = x * y
    #print(z)
    z = x ** y
    #print(z)
    z = x / y
    print(z)
    z = x == y
    print(z)
    z = x != y
    print(z)
    z = -x
    print(z)
    eval(repr(x))