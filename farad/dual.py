"""Dual number implementation for Python forward AD mode.

Author: Matthew Stewart
Date Created: October 30th, 2020
Date Last Updated: November 15th, 2020

This module contains dunder methods to overload built-in Python operators
such as addition and multiplication. This system will only be used for the
forward AD mode, which will be called by the FaradObject.forward() method. If
the method fails to implement dual numbers, it will fall back to using standard
methods.
"""

import numpy as np
import numbers
import reprlib
from typing import NoReturn, List, Union, Optional, Type
Array = Union[List[float], np.ndarray]


class Dual():


    def __init__(self, val: Type[numbers.Integral], der: Optional[Type[Array]] = 1) -> NoReturn:
        """Initialize value and derivatives when called."""
        self._val = val
        self._der = np.array(der)


    @property
    def der(self) -> Type[Array]:
        return self._der  # limits user interference with 'der' attribute


    @property
    def val(self) -> Type[numbers.Integral]:
        return self._val  # limits user interference with 'val' attribute


    def __add__(self, x: Type[Union["Dual", float]]) -> Type["Dual"]:
        """Overload the addition operator (+) to handle Dual class."""
        try:  # try-except loop for Python principle EAFP
            return Dual(self._val + x._val, self._der + x._der)
        except AttributeError:  # if input is not a Dual object but a float
            return Dual(self._val + x, self._der + x)


    def __radd__(self, x: Type[Union["Dual", float]]) -> Type["Dual"]:
        """Revert to __add__ dunder method to handle input reversal."""
        return self.__add__(x)  # operation is commutative


    def __sub__(self, x: Type[Union["Dual", float]]) -> Type["Dual"]:
        """Overload the subtraction operator (-) to handle Dual class."""
        try:
            return Dual(self._val - x._val, self._der - x._der)
        except AttributeError:
            return Dual(self._val - x, self._der - x)


    def __rsub__(self, x: Type[Union["Dual", float]]) -> Type["Dual"]:
        """Revert to __sub__ dunder method to handle input reversal."""
        try:  # operation is not commutative
            return Dual(x._val - self._val, x._der - self._der)
        except AttributeError:
            return Dual(x - self._val, x - self._der)


    def __mul__(self, x: Type[Union["Dual", float]]) -> Type["Dual"]:
        """Overload the multiplication operator (*) to handle Dual class."""
        try:
            return Dual(self._val * x._val, self._der * x._val + self._val * x._der)  # chain rule for derivative
        except AttributeError:
            return Dual(self._val * x, self._der * x)


    def __rmul__(self, x: Type[Union["Dual", float]]) -> Type["Dual"]:
        """Revert to __mul__ dunder method to handle input reversal."""
        return self.__mul__(x)  # operation is commutative


    def __pow__(self, x: Type[Union["Dual", float]]) -> Type["Dual"]:
        """Overload the exponent operator (**) to handle Dual class."""
        try:
            return Dual(self._val**x._val, self._val**x._val*(self._der*(x._val/self._val) + x._der*np.log(self._val)))
        except AttributeError:
            return Dual(self._val**x, self._val**(x-1) * x * self._der)


    def __rpow__(self, x: Type[Union["Dual", float]]) -> Type["Dual"]:
        """Overload input reversed exponent operator to handle Dual class."""
        # Cannot revert to __pow__ dunder method due to non-commutativity of exponent operator
        try: 
            return Dual(x**self._val, x._val**self._val * np.log(x._val) * self._der)
        except AttributeError:
            return Dual(x**self._val, x**self._val * np.log(x) * self._der)


    def __truediv__(self, x: Type[Union["Dual", float]]) -> Type["Dual"]:
        """Overload the division operator (/) to handle Dual class."""
        try:
            return Dual(self._val/x._val, (self._der * x._val - self._val * x._der)/x._val**2)
        except AttributeError:
            return Dual(self._val/x, self._der/x)


    def __rtruediv__(self, x: Type[Union["Dual", float]]) -> Type["Dual"]:
        """Overload input reversed division operator to handle Dual class."""
        # Cannot revert to __truediv__ dunder method due to non-commutativity of divison operator
        return Dual(x/self._val, -x/self._der*self._val**2)


    def __neg__(self: Type[Union["Dual", float]]) -> Type["Dual"]:
        """Overload the unary negation operator (e.g., -x) to handle Dual class."""
        return Dual(-self._val, -self._der)


    def __pos__(self: Type[Union["Dual", float]]) -> Type["Dual"]:
        """Overload the unary positive operator (e.g., +x) to handle Dual class."""
        return Dual(self._val, self._der)


    def __eq__(self, x: Type[Union["Dual", float]]) -> Type[bool]:
        """Overload the equality operator (e.g., x==y) to handle Dual class."""
        try:  # classes equivalent if values and derivatives are equal
            return (self._val == x._val and np.array_equal(self._der, x._der))
        except AttributeError:
            return (self._val == x)


    def __ne__(self, x: Type[Union["Dual", float]]) -> Type[bool]:
        """Overload the inequality operator (e.g., x!=y) to handle Dual class."""
        try:
            return (self._val != x._val or (np.array_equal(self._der, x._der) == False))
        except AttributeError:
            return (self._val != x)


    def __lt__(self, x: Type[Union["Dual", float]]) -> Type[bool]:
        """Overload the less than dunder method (e.g., x<y) to handle Dual class."""
        try:
            return (self._val < x._val)
        except AttributeError:
            return (self._val < x)


    def __le__(self, x: Type[Union["Dual", float]]) -> Type[bool]:
        """Overload the less than or equal to dunder method (e.g., x<=y) to handle Dual class."""
        try:
            return (self._val <= x._val)
        except AttributeError:
            return (self._val <= x)


    def __gt__(self, x: Type[Union["Dual", float]]) -> Type[bool]:
        """Overload the greater than dunder method (e.g., x>y) to handle Dual class."""
        try:
            return (self._val > x._val)
        except AttributeError:
            return (self._val > x)


    def __ge__(self, x: Type[Union["Dual", float]]) -> Type[bool]:
        """Overload the greater than or equal to dunder method (e.g., x>=y) to handle Dual class."""
        try:
            return (self._val >= x._val)
        except AttributeError:
            return (self._val >= x)


    def __repr__(self) -> Type[str]:
        """Prints class definition with inputs - the output can be passed to eval() to instantiate new instance of class Dual."""
        try:
            return f"Dual({reprlib.repr(self._val)},{reprlib.repr(list(self._der))})"
        except TypeError:  # if der attribute cannot be written to a list
            return f"Dual({reprlib.repr(self._val)},{reprlib.repr(self._der)})"

    def __str__(self) -> Type[str]:
        """Prints user-friendly class definition."""
        try:
            return f"Forward-mode {self.__class__.__name__} Object ( Values: {reprlib.repr(np.round(self._val, 4))}, Derivatives: {reprlib.repr(list(np.round(self.der, 4)))} )"
        except TypeError:  # if der attribute cannot be written to a list
            return f"Forward-mode {self.__class__.__name__} Object ( Values: {reprlib.repr(np.round(self._val, 4))}, Derivatives: {reprlib.repr(np.round(self.der, 4))} )"

    def __len__(self) -> Type[str]:
        """Returns length of input vector."""
        try:
            return len(self._val)
        except TypeError:
            return 1


    def __bool__(self) -> Type[bool]:
        return False
