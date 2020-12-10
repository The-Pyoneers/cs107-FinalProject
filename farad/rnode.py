import numpy as np
import numbers
import reprlib
from typing import NoReturn, List, Union, Optional, Type
Array = Union[List[float], np.ndarray, numbers.Integral]


class Rnode:

    def __init__(self, value: numbers.Integral) -> "Rnode":
        """Constructor for Rnode Object class.

        Parameters
        ==========
        value : int/float
            Value of farad.rnode.Rnode object.

        Returns
        =======
        self : Rnode class object
            Object containing value, chilren, and grad_value and derivative attributes.
        chilren attribute store the children of this Rnode object and the derivative
        relationship between this object and its children. grad_value is used to recursively
        calculate the derivative in the reverse mode. Object also has includes overloaded operator methods for custom functionality.

        """
        self.value = value
        self.children = []
        self.grad_value = None

    def clear(self):
        """Function to clear some attributes a Rnode object before reusing as an input to a new function.

        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.

        Returns
        =======
        No returns.

        Notes
        =====
        This function exists to prevent error in calculating reverse mode derivative.
        Mistakes could happen when the same input Rnode x is used for multiple functions.
        In order to reuse x for a new function f(x), this method should be called first
        to clean the chilren and grad_value attributes.

        """
        # to clean some attributes a Rnode object before reusing as an input to a new function
        self.children = []
        self.grad_value = None

    def grad(self):
        """return the function's derivative with respect to the current (self) node through
        recursively reverse calculation.


        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.

        Returns
        =======
        string : float
            Returns the function's derivative with respect to the current node

        Notes
        =====
        The grad_value attribute of the function output node needs to be assigned to 1
        before calling this method to get the right derivative.

        Example
        =======
        >>> x = Rnode(0.11)
        >>> y = x**2
        >>> y.grad_value = 1.0
        >>> x.grad()  # dy/dx
        0.22
        >>> x.clear()
        >>> y1 = x**2 + x
        >>> y1.grad_value = 1.0
        >>> x.grad()  # dy1/dx
        1.22
        """
        # recurse only if the value is not yet cached
        if self.grad_value is None:
            # calculate derivative using chain rule
            self.grad_value = sum(weight * var.grad()
                                  for weight, var in self.children)
        return self.grad_value

    def __add__(self, x: Union["Rnode", int, float]) -> "Rnode":
        """Overload the addition operator (+) to handle Rnode class.

        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.
        x : Rnode object/float/int
            Either class object of type 'Rnode' or int/float value.

        Returns
        =======
        Rnode : Rnode class object
            Returns Rnode object class with updated value and derivative attributes.

        Notes
        =====
        This function overloads the built-in addition operator between Rnode class objects.
        Functionality also exists to succintly support addition with integers or floats to Rnode
        objects by making use of the EAFP (easier to ask forgiveness than permission) in the
        Zen of Python.

        Example
        =======
        >>> Rnode(1.0) + Rnode(2.0)
        Rnode(3.0)
        """
        try:
            z = Rnode(self.value + x.value)
            self.children.append((1., z))  # weight = ∂z/∂self = x.value
            x.children.append((1., z))  # weight = ∂z/∂x = self.value
        except AttributeError:
            z = Rnode(self.value + x)
            self.children.append((1., z))
        return z

    def __radd__(self, x: Union["Rnode", float]) -> "Rnode":
        """Revert to __add__ dunder method to handle input reversal for
        addition operator.

        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.
        x : Rnode object/float/int
            Either class object of type 'Rnode' or int/float value.

        Returns
        =======
        Rnode : Rnode class object
            Returns Rnode object class with updated value and derivative attributes.

        Notes
        =====
        This function overloads the built-in reversed addition operator between Rnode class objects.

        Example
        =======
        >>> 4 + Rnode(2.0)
        Rnode(6.0)
        """
        return self.__add__(x)

    def __sub__(self, x: Union["Rnode", float]) -> "Rnode":
        """Overload the subtraction operator (-) to handle Rnode class.

        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.
        x : Rnode object/float/int
            Either class object of type 'Rnode' or int/float value.

        Returns
        =======
        Rnode : Rnode class object
            Returns Rnode object class with updated value and derivative attributes.

        Notes
        =====
        This function overloads the built-in subtraction operator between Rnode class objects.
        Functionality also exists to succintly support subtraction with integers or floats to Rnode
        objects by making use of the EAFP (easier to ask forgiveness than permission) in the
        Zen of Python.

        Examples
        ========
        >>> Rnode(2.0) - Rnode(1.0)
        Rnode(1.0)
        >>> Rnode(2.0) - 4
        Rnode(-2.0)
        """
        try:
            z = Rnode(self.value - x.value)
            self.children.append((1., z))
            x.children.append((-1., z))
        except AttributeError:
            z = Rnode(self.value - x)
            self.children.append((1., z))
        return z

    def __rsub__(self, x: Union["Rnode", float]) -> "Rnode":
        """Revert to __sub__ dunder method to handle input reversal of
        subtraction method.

        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.
        x : Rnode object/float/int
            Either class object of type 'Rnode' or int/float value.

        Returns
        =======
        Rnode : Rnode class object
            Returns Rnode object class with updated value and derivative attributes.

        Notes
        =====
        This function overloads the built-in reversed subtraction operator between Rnode class objects.

        Examples
        ========
        >>> 4 - Rnode(2.0)
        Rnode(-2.0)
        """
        return self.__sub__(x)

    def __mul__(self, x: Union["Rnode", float]) -> "Rnode":
        """Overload the multiplication operator (*) to handle Rnode class.

        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.
        x : Rnode object/float/int
            Either class object of type 'Rnode' or int/float value.

        Returns
        =======
        Rnode : Rnode class object
            Returns Rnode object class with updated value and derivative attributes.

        Notes
        =====
        This function overloads the built-in multiplication operator between Rnode class objects.
        Functionality also exists to succintly support multiplication with integers or floats to Rnode
        objects by making use of the EAFP (easier to ask forgiveness than permission) in the
        Zen of Python.

        Examples
        ========
        >>> Rnode(2) * Rnode(3)
        Rnode(6)
        >>> Rnode(2) * 3
        Rnode(6)
        """
        try:
            z = Rnode(self.value * x.value)
            self.children.append((x.value, z))
            x.children.append((self.value, z))
        except AttributeError:
            z = Rnode(self.value * x)
            self.children.append((x, z))
        return z

    def __rmul__(self, x: Union["Rnode", int, float]) -> "Rnode":
        """Revert to __mul__ dunder method to handle input reversal.

        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.
        x : Rnode object/float/int
            Either class object of type 'Rnode' or int/float value.

        Returns
        =======
        Rnode : Rnode class object
            Returns Rnode object class with updated value and derivative attributes.

        Notes
        =====
        This function overloads the built-in reversed multiplication operator between Rnode class objects.

        Examples
        ========
        >>> 3 * Rnode(2.0)
        Rnode(6.0)
        """
        return self.__mul__(x)

    def __pow__(self, x: Union["Rnode", int, float]) -> "Rnode":
        """Overload the exponent operator (**) to handle Rnode class.

        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.
        x : Rnode object/float/int
            Either class object of type 'Rnode' or int/float value.

        Returns
        =======
        Rnode : Rnode class object
            Returns Rnode object class with updated value and derivative attributes.

        Notes
        =====
        This function overloads the built-in power operator between Rnode class objects.
        Functionality also exists to succintly support power operations with integers or floats
        by making use of the EAFP (easier to ask forgiveness than permission) in the
        Zen of Python.

        Examples
        ========
        >>> Rnode(1.0) ** 2
        Rnode(1.0)
        >>> Rnode(2.0) ** Rnode(4.0)
        Rnode(16.0)
        """
        try:
            z = Rnode(self.value ** x.value)
            self.children.append((x.value * self.value ** (x.value - 1.), z))
            x.children.append((self.value ** x.value * np.log(self.value), z))
        except AttributeError:
            z = Rnode(self.value ** x)
            self.children.append((x* self.value ** (x - 1.), z))
        return z

    def __rpow__(self, x: Union["Rnode", int, float]) -> "Rnode":
        """Overload input reversed exponent operator to handle Rnode class.

        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.
        x : Rnode object/float/int
            Either class object of type 'Rnode' or int/float value.

        Returns
        =======
        Rnode : Rnode class object
            Returns Rnode object class with updated value and derivative attributes.

        Notes
        =====
        This function overloads the built-in reversed power operator between Rnode class objects.

        Examples
        ========
        >>> 2 ** Rnode(2.0)
        Rnode(4.0)
        """
        # try:
        #     z = Rnode(x.value ** self.value)
        #     x.children.append((self.value * x.value ** (self.value - 1.), z))
        #     self.children.append((x.value ** self.value * np.log(x.value), z))
        # except AttributeError:
        z = Rnode(x ** self.value)
        self.children.append((x ** self.value * np.log(x), z))
        return z

    def __truediv__(self, x: Union["Rnode", int, float]) -> "Rnode":
        """Overload the division operator (/) to handle Rnode class.

        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.
        x : Rnode object/float/int
            Either class object of type 'Rnode' or int/float value.

        Returns
        =======
        Rnode : Rnode class object
            Returns Rnode object class with updated value and derivative attributes.

        Notes
        =====
        This function overloads the built-in division operator between Rnode class objects.
        Functionality also exists to succintly support divison with integers or floats to Rnode
        objects by making use of the EAFP (easier to ask forgiveness than permission) in the
        Zen of Python.

        Examples
        ========
        >>> Rnode(2.0) / Rnode(1.0)
        Rnode(2.0)
        >>> Rnode(2.0) / 4
        Rnode(0.5)
        """
        try:
            z = Rnode(self.value / x.value)
            self.children.append((1./x.value, z))
            x.children.append((- self.value / (x.value)**2, z))
        except AttributeError:
            z = Rnode(self.value / x)
            self.children.append((1./x, z))
        return z

    def __rtruediv__(self, x: Union["Rnode", int, float]) -> "Rnode":
        """Overload input reversed division operator to handle Rnode class.

        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.
        x : Rnode object/float/int
            Either class object of type 'Rnode' or int/float value.

        Returns
        =======
        Rnode : Rnode class object
            Returns Rnode object class with updated value and derivative attributes.

        Notes
        =====
        This function overloads the built-in reversed division operator between Rnode class objects.


        Examples
        ========
        >>> 3 / Rnode(1.0)
        Rnode(3.0)
        """
        # try:
        #     z = Rnode(x.value / self.value)
        #     x.children.append((1./self.value, z))
        #     self.children.append((- x.value / (self.value)**2, z))
        # except AttributeError:
        z = Rnode(x / self.value)
        self.children.append((- x / (self.value)**2, z))
        return z

    def __neg__(self: Union["Rnode", int, float]) -> "Rnode":
        """Overload the unary negation operator (e.g., -x) to handle Rnode class.

        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.
        x : Rnode object/float/int
            Either class object of type 'Rnode' or int/float value.

        Returns
        =======
        Rnode : Rnode class object
            Returns Rnode object class with updated value and derivative attributes.

        Notes
        =====
        Called by using the standard unary positive operator (+x) between two
        Rnode objects or a Rnode object and an integer/float value.

        Examples
        ========
        >>> -Rnode(1.0)
        Rnode(-1.0)
        """
        z = Rnode(-self.value)
        self.children.append((-1, z))
        return z

    def __pos__(self: Union["Rnode", int, float]) -> "Rnode":
        """Overload the unary positive operator (e.g., +x) to handle Rnode class.

        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.
        x : Rnode object/float/int
            Either class object of type 'Rnode' or int/float value.

        Returns
        =======
        Rnode : Rnode class object
            Returns Rnode object class with updated value and derivative attributes.

        Notes
        =====
        Called by using the standard unary positive operator (+x) between two
        Rnode objects or a Rnode object and an integer/float value.

        Examples
        ========
        >>> +Rnode(1.0)
        Rnode(1.0)
        """
        z = Rnode(self.value)
        self.children.append((1, z))
        return z

    def __eq__(self, x: Union["Rnode", int, float]) -> bool:
        """Overload the equality operator (e.g., x==y) to handle Rnode class.

        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.
        x : Rnode object/float/int
            Either class object of type 'Rnode' or int/float value.

        Returns
        =======
        Boolean expression : bool
            Returns True if the value attribute of self is equal to the
            passed value (if int/float) or is equal to the value attribute of the
            passed Rnode class object.

        Notes
        =====
        Called by using the standard equality operator (==) between two
        Rnode objects or a Rnode object and an integer/float value.

        Examples
        ========
        >>> Rnode(1.0) == Rnode(2.0)
        False
        >>> Rnode(1.0) == Rnode(1.0)
        True
        """
        try:  # classes equivalent if values and derivatives are equal
            return (self.value == x.value and self.grad_value == x.grad_value)
        except AttributeError:
            return (self.value == x)

    def __ne__(self, x: Union["Rnode", int, float]) -> bool:
        """Overload the inequality operator (e.g., x!=y) to handle Rnode class.

        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.
        x : Rnode object/float/int
            Either class object of type 'Rnode' or int/float value.

        Returns
        =======
        Boolean expression : bool
            Returns True if the value attribute of self is not equal to the
            passed value (if int/float) or is not equal to the value attribute of the
            passed Rnode class object.

        Notes
        =====
        Called by using the standard inequality operator (!=) between two
        Rnode objects or a Rnode object and an integer/float value.

        Examples
        ========
        >>> Rnode(1.0) != Rnode(2.0)
        True
        >>> Rnode(1.0) != Rnode(1.0)
        False
        """
        try:
            return (self.value != x.value or self.grad_value != x.grad_value)
        except AttributeError:
            return (self.value != x)

    def __lt__(self, x: Union["Rnode", int, float]) -> bool:
        """Overload the less than dunder method (e.g., x<y) to handle Rnode class.

        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.
        x : Rnode object/float/int
            Either class object of type 'Rnode' or int/float value.

        Returns
        =======
        Boolean expression : bool
            Returns True if the value attribute of self is smaller than the
            passed value (if int/float) or is smaller than the value attribute of the
            passed Rnode class object.

        Notes
        =====
        Called by using the standard less than operator (<) between two
        Rnode objects or a Rnode object and an integer/float value.

        Examples
        ========
        >>> Rnode(1.0) < Rnode(2.0)
        True
        >>> Rnode(1.0) < Rnode(1.0)
        False
        """
        try:
            return (self.value < x.value)
        except AttributeError:
            return (self.value < x)

    def __le__(self, x: Union["Rnode", int, float]) -> bool:
        """Overload the less than or equal to dunder method (e.g., x<=y) to handle Rnode class.

        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.
        x : Rnode object/float/int
            Either class object of type 'Rnode' or int/float value.

        Returns
        =======
        Boolean expression : bool
            Returns True if the value attribute of self is smaller than or equal to the
            passed value (if int/float) or is smaller than or equal to the value attribute
            of the passed Rnode class object.

        Notes
        =====
        Called by using the standard less than or equal to operator (<=) between two
        Rnode objects or a Rnode object and an integer/float value.

        Examples
        ========
        >>> Rnode(1.0) <= Rnode(2.0)
        True
        >>> Rnode(1.0) <= Rnode(1.0)
        True
        >>> Rnode(2.0) <= Rnode(1.0)
        False
        """
        try:
            return (self.value <= x.value)
        except AttributeError:
            return (self.value <= x)

    def __gt__(self, x: Union["Rnode", int, float]) -> bool:
        """Overload the greater than dunder method (e.g., x>y) to handle Rnode class.

        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.
        x : Rnode object/float/int
            Either class object of type 'Rnode' or int/float value.

        Returns
        =======
        Boolean expression : bool
            Returns True if the value attribute of self is larger than the
            passed value (if int/float) or is larger than the value attribute of the
            passed Rnode class object.

        Notes
        =====
        Called by using the standard greater than operator (>) between two
        Rnode objects or a Rnode object and an integer/float value.

        Examples
        ========
        >>> Rnode(3.0) > Rnode(2.0)
        True
        >>> Rnode(1.0) > Rnode(1.0)
        False
        """
        try:
            return (self.value > x.value)
        except AttributeError:
            return (self.value > x)

    def __ge__(self, x: Union["Rnode", int, float]) -> bool:
        """Overload the greater than or equal to dunder method (e.g., x>=y) to handle Rnode class.

        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.
        x : Rnode object/float/int
            Either class object of type 'Rnode' or int/float value.

        Returns
        =======
        Boolean expression : bool
            Returns True if the value attribute of self is larger than or equal to the
            passed value (if int/float) or is larger than the value attribute of the
            passed Rnode class object.

        Notes
        =====
        Called by using the standard greater than or equal to operator (>=) between two
        Rnode objects or a Rnode object and an integer/float value.

        Examples
        ========
        >>> Rnode(3.0) >= Rnode(2.0)
        True
        >>> Rnode(1.0) >= Rnode(1.0)
        True
        >>> Rnode(-2.0) >= Rnode(1.0)
        False
        """
        try:
            return (self.value >= x.value)
        except AttributeError:
            return (self.value >= x)

    def __repr__(self) -> str:
        """Prints class definition with inputs - the output can be passed to eval()
        to instantiate new instance of class Rnode.

        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.

        Returns
        =======
        string : str
            Returns description of Rnode class object compatible with eval() built-in method.

        Notes
        =====
        Reprlib is used to limit potential parameters passed for large Rnode objects.

        Example
        =======
        >>> repr(Rnode(1.0))
        'Rnode(1.0)'
        """

        return f"Rnode({reprlib.repr(self.value)})"

    def __str__(self) -> str:
        """Prints user-friendly class definition.

        Parameters
        ==========
        self : Rnode class object
            Class object of type 'Rnode'.

        Returns
        =======
        string : str
            Returns description of Rnode class object.

        Notes
        =====
        Reprlib is used to limit potential parameters passed for large Rnode objects.

        Example
        =======
        >>> str(Rnode(1.0))
        'Reverse-mode Rnode Object ( Values: 1.0 )'
        """

        return f"Reverse-mode {self.__class__.__name__} Object ( Values: {reprlib.repr(np.round(self.value, 4))} )"
