"""Dual number implementation for Python forward AD mode.

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
Array = Union[List[float], np.ndarray, numbers.Integral]


class Dual:


    def __init__(self, val: numbers.Integral, der: Optional[Array] = 1) -> "Dual":
        """Constructor for Dual Object class.

        Parameters
        ==========
        val : int/float
            Value of farad.dual.Dual object.
        der : optional[list[float],int/float, np.ndarray]
            First derivative of farad.Dual.object.

        Returns
        =======
        self : Dual class object
            Object containing value and derivative attributes. Object also
        has includes overloaded operator methods for custom functionality.

        Notes
        =====
        Currently, only first order derivatives of scalar functions
        are supported. This will be extended in later versions to handle
        vector inputs with jacobians.
        """
        self._val = val
        self._der = der


    @property
    def der(self) -> Array:
        """Function to allow access to private first derivative attribute.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.

        Returns
        =======
        der : optional[list[float],int/float, np.ndarray]
            Stored private variable for first derivative of dual class object.

        Notes
        =====
        This function exists to prevent the user from directly interacting with
        the derivative attribute. Instead, the user can interact safely using the
        'der' method. The property decorator is added in case future functionality
        is added to manipulate the stored private variables to ensure changes are
        propagated throughout the class.

        Example
        =======
        >>> Dual(1.0,1.0).der
        1.0
        """
        return self._der  # limits user interference with 'der' attribute


    @property
    def val(self) -> numbers.Integral:
        """Function to allow access to private value attribute.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.

        Returns
        =======
        der : int/float
            Stored private variable for value attribute of dual class object.

        Notes
        =====
        This function exists to prevent the user from directly interacting with
        the value attribute. Instead, the user can interact safely using the
        'val' method. The property decorator is added in case future functionality
        is added to manipulate the stored private variables to ensure changes are
        propagated throughout the class.

        Example
        =======
        >>> Dual(1.0,1.0).val
        1.0
        """
        return self._val  # limits user interference with 'val' attribute


    def __add__(self, x: Union["Dual", int, float]) -> "Dual":
        """Overload the addition operator (+) to handle Dual class.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.
        x : Dual object/float/int
            Either class object of type 'Dual' or int/float value.

        Returns
        =======
        Dual : Dual class object
            Returns dual object class with updated value and derivative attributes.

        Notes
        =====
        This function overloads the built-in addition operator between Dual class objects.
        Functionality also exists to succintly support addition with integers or floats to Dual
        objects by making use of the EAFP (easier to ask forgiveness than permission) in the
        Zen of Python.

        Example
        =======
        >>> Dual(1.0,4.0) + Dual(2.0,3.0)
        Dual(3.0,7.0)
        """
        try:  # try-except loop for Python principle EAFP
            return Dual(self._val + x._val, self._der + x._der)
        except AttributeError:  # if input is not a Dual object but a float
            return Dual(self._val + x, self._der)


    def __radd__(self, x: Union["Dual", float]) -> "Dual":
        """Revert to __add__ dunder method to handle input reversal for
        addition operator.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.
        x : Dual object/float/int
            Either class object of type 'Dual' or int/float value.

        Returns
        =======
        Dual : Dual class object
            Returns dual object class with updated value and derivative attributes.

        Notes
        =====
        This function overloads the built-in reversed addition operator between Dual class objects.

        Example
        =======
        >>> 4 + Dual(2.0,3.0)
        Dual(6.0,3.0)
        """
        return self.__add__(x)  # operation is commutative


    def __sub__(self, x: Union["Dual", float]) -> "Dual":
        """Overload the subtraction operator (-) to handle Dual class.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.
        x : Dual object/float/int
            Either class object of type 'Dual' or int/float value.

        Returns
        =======
        Dual : Dual class object
            Returns dual object class with updated value and derivative attributes.

        Notes
        =====
        This function overloads the built-in subtraction operator between Dual class objects.
        Functionality also exists to succintly support subtraction with integers or floats to Dual
        objects by making use of the EAFP (easier to ask forgiveness than permission) in the
        Zen of Python.

        Examples
        ========
        >>> Dual(2.0, 3) - Dual(1.0, 2)
        Dual(1.0,1)
        >>> Dual(2.0, 3) - 4
        Dual(-2.0,-1)
        """
        try:
            return Dual(self._val - x._val, np.asarray(self._der) - np.asarray(x._der))
        except AttributeError:
            return Dual(self._val - x, self._der - x)


    def __rsub__(self, x: Union["Dual", float]) -> "Dual":
        """Revert to __sub__ dunder method to handle input reversal of
        subtraction method.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.
        x : Dual object/float/int
            Either class object of type 'Dual' or int/float value.

        Returns
        =======
        Dual : Dual class object
            Returns dual object class with updated value and derivative attributes.

        Notes
        =====
        This function overloads the built-in reversed subtraction operator between Dual class objects.

        Examples
        ========
        >>> Dual(2.0, 3) - Dual(1.0, 2)
        Dual(1.0,1)
        >>> Dual(2.0, 3) - 4
        Dual(-2.0,-1)
        """
        try:  # operation is not commutative
            return Dual(x._val - self._val, x._der - self._der)
        except AttributeError:
            return Dual(x - self._val, x - self._der)


    def __mul__(self, x: Union["Dual", float]) -> "Dual":
        """Overload the multiplication operator (*) to handle Dual class.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.
        x : Dual object/float/int
            Either class object of type 'Dual' or int/float value.

        Returns
        =======
        Dual : Dual class object
            Returns dual object class with updated value and derivative attributes.

        Notes
        =====
        This function overloads the built-in multiplication operator between Dual class objects.
        Functionality also exists to succintly support multiplication with integers or floats to Dual
        objects by making use of the EAFP (easier to ask forgiveness than permission) in the
        Zen of Python.

        Examples
        ========
        >>> Dual(2,3) * Dual(1,2)
        Dual(2,7)
        >>> Dual(2,3) * 3
        Dual(6,9)
        """
        try:
            return Dual(self._val * x._val, self._der * x._val + self._val * x._der)  # chain rule for derivative
        except AttributeError:
            return Dual(self._val * x, self._der * x)


    def __rmul__(self, x: Union["Dual", int, float]) -> "Dual":
        """Revert to __mul__ dunder method to handle input reversal.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.
        x : Dual object/float/int
            Either class object of type 'Dual' or int/float value.

        Returns
        =======
        Dual : Dual class object
            Returns dual object class with updated value and derivative attributes.

        Notes
        =====
        This function overloads the built-in reversed multiplication operator between Dual class objects.

        Examples
        ========
        >>> 3 * Dual(1.0, 2)
        Dual(3.0,6)
        """
        return self.__mul__(x)  # operation is commutative


    def __pow__(self, x: Union["Dual", int, float]) -> "Dual":
        """Overload the exponent operator (**) to handle Dual class.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.
        x : Dual object/float/int
            Either class object of type 'Dual' or int/float value.

        Returns
        =======
        Dual : Dual class object
            Returns dual object class with updated value and derivative attributes.

        Notes
        =====
        This function overloads the built-in power operator between Dual class objects.
        Functionality also exists to succintly support power operations with integers or floats
        by making use of the EAFP (easier to ask forgiveness than permission) in the
        Zen of Python.

        Examples
        ========
        >>> Dual(1.0, 2) ** 2
        Dual(1.0,4.0)
        >>> Dual(1.0, 3.0) ** Dual(4.0, 5.0)
        Dual(1.0,12.0)
        """
        try:
            return Dual(self._val**x._val, self._val**x._val*(self._der*(x._val/self._val) + x._der*np.log(self._val)))
        except AttributeError:
            return Dual(self._val**x, self._val**(x-1) * x * np.asarray(self._der))


    def __rpow__(self, x: Union["Dual", int, float]) -> "Dual":
        """Overload input reversed exponent operator to handle Dual class.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.
        x : Dual object/float/int
            Either class object of type 'Dual' or int/float value.

        Returns
        =======
        Dual : Dual class object
            Returns dual object class with updated value and derivative attributes.

        Notes
        =====
        This function overloads the built-in reversed power operator between Dual class objects.

        Examples
        ========
        >>> 2 ** Dual(1.0,2)
        Dual(2.0,2.772588722239781)
        """
        # Cannot revert to __pow__ dunder method due to non-commutativity of exponent operator
        try:
            return Dual(x**self._val, x._val**self._val * np.log(x._val) * self._der)
        except AttributeError:
            return Dual(x**self._val, x**self._val * np.log(x) * self._der)


    def __truediv__(self, x: Union["Dual", int, float]) -> "Dual":
        """Overload the division operator (/) to handle Dual class.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.
        x : Dual object/float/int
            Either class object of type 'Dual' or int/float value.

        Returns
        =======
        Dual : Dual class object
            Returns dual object class with updated value and derivative attributes.

        Notes
        =====
        This function overloads the built-in division operator between Dual class objects.
        Functionality also exists to succintly support divison with integers or floats to Dual
        objects by making use of the EAFP (easier to ask forgiveness than permission) in the
        Zen of Python.

        Examples
        ========
        >>> Dual(2.0,3.0) / Dual(1.0,4.0)
        Dual(2.0,-5.0)
        >>> Dual(2.0,3.0) / 4
        Dual(0.5,0.75)
        """
        try:
            return Dual(self._val/x._val, (self._der * x._val - self._val * x._der)/x._val**2)
        except AttributeError:
            return Dual(self._val/x, self._der/x)


    def __rtruediv__(self, x: Union["Dual", int, float]) -> "Dual":
        """Overload input reversed division operator to handle Dual class.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.
        x : Dual object/float/int
            Either class object of type 'Dual' or int/float value.

        Returns
        =======
        Dual : Dual class object
            Returns dual object class with updated value and derivative attributes.

        Notes
        =====
        This function overloads the built-in reversed division operator between Dual class objects.


        Examples
        ========
        >>> 3 / Dual(1.0,4.0)
        Dual(3.0,-12.0)
        """
        # Cannot revert to __truediv__ dunder method due to non-commutativity of divison operator
        return Dual(x/self._val, -(x/self._val**2) * self._der)


    def __neg__(self: Union["Dual", int, float]) -> "Dual":
        """Overload the unary negation operator (e.g., -x) to handle Dual class.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.
        x : Dual object/float/int
            Either class object of type 'Dual' or int/float value.

        Returns
        =======
        Dual : Dual class object
            Returns dual object class with updated value and derivative attributes.

        Notes
        =====
        Called by using the standard unary positive operator (+x) between two
        dual objects or a dual object and an integer/float value.

        Examples
        ========
        >>> -Dual(1.0,4.0)
        Dual(-1.0,-4.0)
        """
        return Dual(-self._val, -self._der)


    def __pos__(self: Union["Dual", int, float]) -> "Dual":
        """Overload the unary positive operator (e.g., +x) to handle Dual class.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.
        x : Dual object/float/int
            Either class object of type 'Dual' or int/float value.

        Returns
        =======
        Dual : Dual class object
            Returns dual object class with updated value and derivative attributes.

        Notes
        =====
        Called by using the standard unary positive operator (+x) between two
        dual objects or a dual object and an integer/float value.

        Examples
        ========
        >>> +Dual(1.0,4.0)
        Dual(1.0,4.0)
        """
        return Dual(self._val, self._der)


    def __eq__(self, x: Union["Dual", int, float]) -> bool:
        """Overload the equality operator (e.g., x==y) to handle Dual class.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.
        x : Dual object/float/int
            Either class object of type 'Dual' or int/float value.

        Returns
        =======
        Boolean expression : bool
            Returns True if the value attribute of self is equal to the
            passed value (if int/float) or is equal to the value attribute of the
            passed Dual class object.

        Notes
        =====
        Called by using the standard equality operator (==) between two
        dual objects or a dual object and an integer/float value.

        Examples
        ========
        >>> Dual(1.0,4.0) == Dual(2.0,3.0)
        False
        >>> Dual(5.0,1.0) == Dual(3.0,1.0)
        False
        >>> Dual(1.0,1.0) == Dual(1.0,1.0)
        True
        >>> Dual(4.0,1.0) == 4.0
        True
        """
        try:  # classes equivalent if values and derivatives are equal
            return (self._val == x._val and np.array_equal(self._der, x._der))
        except AttributeError:
            return (self._val == x)


    def __ne__(self, x: Union["Dual", int, float]) -> bool:
        """Overload the inequality operator (e.g., x!=y) to handle Dual class.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.
        x : Dual object/float/int
            Either class object of type 'Dual' or int/float value.

        Returns
        =======
        Boolean expression : bool
            Returns True if the value attribute of self is not equal to the
            passed value (if int/float) or is not equal to the value attribute of the
            passed Dual class object.

        Notes
        =====
        Called by using the standard inequality operator (!=) between two
        dual objects or a dual object and an integer/float value.

        Examples
        ========
        >>> Dual(1.0,4.0) != Dual(2.0,3.0)
        True
        >>> Dual(5.0,1.0) != Dual(3.0,1.0)
        True
        >>> Dual(1.0,1.0) != Dual(1.0,1.0)
        False
        >>> Dual(4.0,1.0) != 4.0
        False
        """
        try:
            return self._val != x._val or (np.array_equal(self._der, x._der) is False)
        except AttributeError:
            return self._val != x


    def __lt__(self, x: Union["Dual", int, float]) -> bool:
        """Overload the less than dunder method (e.g., x<y) to handle Dual class.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.
        x : Dual object/float/int
            Either class object of type 'Dual' or int/float value.

        Returns
        =======
        Boolean expression : bool
            Returns True if the value attribute of self is smaller than the
            passed value (if int/float) or is smaller than the value attribute of the
            passed Dual class object.

        Notes
        =====
        Called by using the standard less than operator (<) between two
        dual objects or a dual object and an integer/float value.

        Examples
        ========
        >>> Dual(1.0,4.0) < Dual(2.0,3.0)
        True
        >>> Dual(5.0,1.0) < Dual(3.0,1.0)
        False
        >>> Dual(1.0,1.0) < Dual(1.0,1.0)
        False
        >>> Dual(4.0,1.0) < 3.0
        False
        """
        try:
            return (self._val < x._val)
        except AttributeError:
            return (self._val < x)


    def __le__(self, x: Union["Dual", int, float]) -> bool:
        """Overload the less than or equal to dunder method (e.g., x<=y) to handle Dual class.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.
        x : Dual object/float/int
            Either class object of type 'Dual' or int/float value.

        Returns
        =======
        Boolean expression : bool
            Returns True if the value attribute of self is smaller than or equal to the
            passed value (if int/float) or is smaller than or equal to the value attribute
            of the passed Dual class object.

        Notes
        =====
        Called by using the standard less than or equal to operator (<=) between two
        dual objects or a dual object and an integer/float value.

        Examples
        ========
        >>> Dual(1.0,4.0) <= Dual(2.0,3.0)
        True
        >>> Dual(5.0,1.0) <= Dual(3.0,1.0)
        False
        >>> Dual(1.0,1.0) <= Dual(1.0,1.0)
        True
        >>> Dual(4.0,1.0) <= 3.0
        False
        """
        try:
            return (self._val <= x._val)
        except AttributeError:
            return (self._val <= x)


    def __gt__(self, x: Union["Dual", int, float]) -> bool:
        """Overload the greater than dunder method (e.g., x>y) to handle Dual class.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.
        x : Dual object/float/int
            Either class object of type 'Dual' or int/float value.

        Returns
        =======
        Boolean expression : bool
            Returns True if the value attribute of self is larger than the
            passed value (if int/float) or is larger than the value attribute of the
            passed Dual class object.

        Notes
        =====
        Called by using the standard greater than operator (>) between two
        dual objects or a dual object and an integer/float value.

        Examples
        ========
        >>> Dual(1.0,4.0) > Dual(2.0,3.0)
        False
        >>> Dual(5.0,1.0) > Dual(3.0,1.0)
        True
        >>> Dual(1.0,1.0) > Dual(1.0,1.0)
        False
        >>> Dual(4.0,1.0) > 3.0
        True
        """
        try:
            return (self._val > x._val)
        except AttributeError:
            return (self._val > x)


    def __ge__(self, x: Union["Dual", int, float]) -> bool:
        """Overload the greater than or equal to dunder method (e.g., x>=y) to handle Dual class.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.
        x : Dual object/float/int
            Either class object of type 'Dual' or int/float value.

        Returns
        =======
        Boolean expression : bool
            Returns True if the value attribute of self is larger than or equal to the
            passed value (if int/float) or is larger than the value attribute of the
            passed Dual class object.

        Notes
        =====
        Called by using the standard greater than or equal to operator (>=) between two
        dual objects or a dual object and an integer/float value.

        Examples
        ========
        >>> Dual(1.0,4.0) >= Dual(2.0,3.0)
        False
        >>> Dual(5.0,1.0) >= Dual(3.0,1.0)
        True
        >>> Dual(1.0,1.0) >= Dual(1.0,1.0)
        True
        >>> Dual(4.0,1.0) >= 3.0
        True
        """
        try:
            return (self._val >= x._val)
        except AttributeError:
            return (self._val >= x)


    def __repr__(self) -> str:
        """Prints class definition with inputs - the output can be passed to eval()
        to instantiate new instance of class Dual.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.

        Returns
        =======
        string : str
            Returns description of Dual class object compatible with eval() built-in method.

        Notes
        =====
        Reprlib is used to limit potential parameters passed for large Dual objects.

        Example
        =======
        >>> repr(Dual(1.0,4.0))
        'Dual(1.0,4.0)'
        """
        try:
            return f"Dual({reprlib.repr(self._val)},{reprlib.repr(list(self._der))})"
        except TypeError:  # if der attribute cannot be written to a list
            return f"Dual({reprlib.repr(self._val)},{reprlib.repr(self._der)})"

    def __str__(self) -> str:
        """Prints user-friendly class definition.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.

        Returns
        =======
        string : str
            Returns description of Dual class object.

        Notes
        =====
        Reprlib is used to limit potential parameters passed for large Dual objects.

        Example
        =======
        >>> str(Dual(1.0,4.0))
        'Forward-mode Dual Object ( Values: 1.0, Derivatives: 4.0 )'
        """
        try:
            return f"Forward-mode {self.__class__.__name__} Object ( Values: {reprlib.repr(np.round(self._val, 4))}, Derivatives: {reprlib.repr(list(np.round(self.der, 4)))} )"
        except TypeError:  # if der attribute cannot be written to a list
            return f"Forward-mode {self.__class__.__name__} Object ( Values: {reprlib.repr(np.round(self._val, 4))}, Derivatives: {reprlib.repr(np.round(self.der, 4))} )"

    def __len__(self) -> int:
        """Returns length of input vector.

        Parameters
        ==========
        self : Dual class object
            Class object of type 'Dual'.

        Returns
        =======
        length : int
            integer output corresponding to 'length' of Dual object.

        Example
        =======
        >>> len(Dual(1.0,4.0))
        1
        """
        try:
            return len(self._val)
        except TypeError:
            return 1
