import numpy as np
import numbers
import reprlib
from typing import NoReturn, List, Union, Optional, Type
Array = Union[List[float], np.ndarray, numbers.Integral]


class Rnode:

    def __init__(self, value):
        self.value = value
        self.children = []
        self.grad_value = None

    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        pass

    def __add__(self, x):
        try:
            z = Rnode(self.value + x.value)
            self.children.append((1., z))  # weight = ∂z/∂self = x.value
            x.children.append((1., z))  # weight = ∂z/∂x = self.value
        except AttributeError:
            z = Rnode(self.value + x)
            self.children.append((1., z))
        return z

    def __radd__(self, x):
        return self.__add__(x)

    def __sub__(self, x):
        try:
            z = Rnode(self.value - x.value)
            self.children.append((1., z))
            x.children.append((-1., z))
        except AttributeError:
            z = Rnode(self.value - x)
            self.children.append((1., z))
        return z

    def __rsub__(self, x):
        return self.__sub__(x)

    def __mul__(self, x):
        try:
            z = Rnode(self.value * x.value)
            self.children.append((x.value, z))
            x.children.append((self.value, z))
        except AttributeError:
            z = Rnode(self.value * x)
            self.children.append((x, z))
        return z

    def __rmul__(self, x):
        return self.__mul__(x)

    def __pow__(self, x):
        try:
            z = Rnode(self.value ** x.value)
            self.children.append((x.value * self.value ** (x.value - 1.), z))
            x.children.append((self.value ** x.value * np.log(self.value), z))
        except AttributeError:
            z = Rnode(self.value ** x)
            self.children.append((x* self.value ** (x - 1.), z))
        return z

    def __rpow__(self, x):
        try:
            z = Rnode(x.value ** self.value)
            x.children.append((self.value * x.value ** (self.value - 1.), z))
            self.children.append((x.value ** self.value * np.log(x.value), z))
        except AttributeError:
            z = Rnode(x ** self.value)
            self.children.append((x ** self.value * np.log(x), z))
        return z

    def __truediv__(self, x):
        try:
            z = Rnode(self.value / x.value)
            self.children.append((1./x.value, z))
            x.children.append((- self.value / (x.value)**2, z))
        except AttributeError:
            z = Rnode(self.value / x)
            self.children.append((1./x, z))
        return z

    def __rtruediv__(self, x):
        try:
            z = Rnode(x.value / self.value)
            x.children.append((1./self.value, z))
            self.children.append((- x.value / (self.value)**2, z))
        except AttributeError:
            z = Rnode(x / self.value)
            self.children.append((- x / (self.value)**2, z))
        return z

    def __neg__(self):
        z = Rnode(-self.value)
        self.children.append(-1, z)
        return z

    def __pos__(self):
        z = Rnode(self.value)
        self.children.append(1, z)
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

    def grad(self):
        # recurse only if the value is not yet cached
        if self.grad_value is None:
            # calculate derivative using chain rule
            self.grad_value = sum(weight * var.grad()
                                  for weight, var in self.children)
        return self.grad_value
