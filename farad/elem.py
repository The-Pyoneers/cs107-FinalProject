"""Math functionality for farad package."""

__all__ = ['sin', 'cos', 'tan', 'log', 'log10', 'sinh', 'cosh', 'tanh', \
           'log2', 'exp', 'sqrt', 'arccos', 'arcsin', 'arctan', \
           'relu', 'logistic', 'relu6']


from farad.dual import Dual
from farad.rnode import Rnode
import numpy as np
from typing import Union, List


def sin(x: Union[Rnode, Dual, float]) -> Union[Rnode, Dual, float, List[float]]:
    """Calculate sine of the input in radians.

    Parameters:
    x : array_like, Rnode object, or Dual Object. Angle, in radians (2 pi rad equals 360 degrees).

    Returns:
    y : array_like, Rnode object, or Dual Object. The sine of each element of x.
    """
    try:
        z = Rnode(np.sin(x.value))
        x.children.append((np.cos(x.value), z))
        return z
    except AttributeError:
        try:  # Python EAFP principle - assume input is type Dual
            return Dual(np.sin(x._val), np.cos(x._val)*np.asarray(x._der))
        except AttributeError:
            return np.sin(x)  # Default to numpy implementation


def cos(x: Union[Rnode, Dual, float]) -> Union[Rnode, Dual, float, List[float]]:
    """Calculate cosine of the input in radians.

    Parameters:
    x : array_like, Rnode object, or Dual Object. Angle, in radians (2 pi rad equals 360 degrees).

    Returns:
    y : array_like, Rnode object, or Dual Object. The cosine of each element of x.
    """
    try:
        z = Rnode(np.cos(x.value))
        x.children.append((-np.sin(x.value), z))
        return z
    except AttributeError:
        try:
            return Dual(np.cos(x._val), -np.sin(x._val)*np.asarray(x._der))
        except AttributeError:
            return np.cos(x)  # Default to numpy implementation


def tan(x: Union[Rnode, Dual, float]) -> Union[Rnode, Dual, float, List[float]]:
    """Calculate tangent of the input in radians.

    Parameters:
    x : array_like, Rnode object, or Dual Object. Angle, in radians (2 pi rad equals 360 degrees).

    Returns:
    y : array_like, Rnode object, or Dual Object. The tangent of each element of x.
    """

    try:
        z = Rnode(np.tan(x.value))
        x.children.append((1 / (np.cos(x.value) ** 2), z))
        return z
    except AttributeError:
        try:
            return Dual(np.tan(x._val), 1/np.cos(x._val)**2*np.asarray(x._der))
        except AttributeError:
            return np.tan(x)  # Default to numpy implementation


def log(x: Union[Rnode, Dual, float]) -> Union[Rnode, Dual, float, List[float]]:
    """Calculate natural logarithm of input.

    Parameters:
    x : array_like, Rnode object, or Dual Object.

    Returns:
    y : array_like, Rnode object, or Dual Object. The natural logarithm of each element of x.
    """
    try:
        z = Rnode(np.log(x.value))
        x.children.append(((1/x.value), z))
        return z
    except AttributeError:
        try:
            if x._val <= 0:
                raise ValueError('Domain of logarithm is {x > 0}')
            return Dual(np.log(x._val), (1/x._val)*np.asarray(x._der))
        except AttributeError:
            return np.log(x)  # Default to numpy implementation


def log10(x: Union[Rnode, Dual, float]) -> Union[Rnode, Dual, float, List[float]]:
    """Calculates the base-10 logarithm of input.

    Parameters:
    x : array_like, Rnode object, or Dual Object.

    Returns:
    y : array_like, Rnode object, or Dual Object. The base-10 logarithm of each element of x.
    """

    try:
        z = Rnode(np.log10(x.value))
        x.children.append((1/(x.value * np.log(10)), z))
        return z
    except AttributeError:
        try:
            if x._val <= 0:
                raise ValueError('Domain of logarithm is {x > 0}')
            return Dual(np.log10(x._val), (1/(x._val*np.log(10)))*np.asarray(x._der))
        except AttributeError:
            return np.log10(x)  # Default to numpy implementation


def log2(x: Union[Rnode, Dual, float]) -> Union[Rnode, Dual, float, List[float]]:
    """Calculates the base-2 logarithm of input.

    Parameters:
    x : array_like, Rnode object, or Dual Object.

    Returns:
    y : array_like, Rnode object, or Dual Object. The base-2 logarithm of each element of x.
    """
    try:
        z = Rnode(np.log2(x.value))
        x.children.append((1/(x.value * np.log(2)), z))
        return z
    except AttributeError:
        try:
            if x._val <= 0:
                raise ValueError('Domain of logarithm is {x > 0}')
            return Dual(np.log2(x._val), (1/(x._val*np.log(2)))*np.asarray(x._der))
        except AttributeError:
            return np.log2(x)  # Default to numpy implementation


def sinh(x: Union[Rnode, Dual, float]) -> Union[Rnode, Dual, float, List[float]]:
    """Calculates the hyberbolic sine of input.

    Parameters:
    x : array_like, Rnode object, or Dual Object.

    Returns:
    y : array_like, Rnode object, or Dual Object. The hyperbolic sine of each element of x.
    """
    try:
        z = Rnode(np.sinh(x.value))
        x.children.append((np.cosh(x.value), z))
        return z
    except AttributeError:
        try:
            return Dual(np.sinh(x.val), np.cosh(x.val) * np.asarray(x.der))
        except AttributeError:
            return np.sinh(x)  # default to numpy implementation


def cosh(x: Union[Rnode, Dual, float]) -> Union[Rnode, Dual, float, List[float]]:
    """Calculates the hyberbolic cosine of input.

    Parameters:
    x : array_like, Rnode object, or Dual Object.

    Returns:
    y : array_like, Rnode object, or Dual Object. The hyperbolic cosine of each element of x.
    """
    try:
        z = Rnode(np.cosh(x.value))
        x.children.append((np.sinh(x.value), z))
        return z
    except AttributeError:
        try:
            return Dual(np.cosh(x.val), np.sinh(x.val) * np.asarray(x.der))
        except AttributeError:
            return np.cosh(x)  # default to numpy implementation


def tanh(x: Union[Rnode, Dual, float]) -> Union[Rnode, Dual, float, List[float]]:
    """Calculates the hyberbolic tangent of input.

    Parameters:
    x : array_like, Rnode object, or Dual Object.

    Returns:
    y : array_like, Rnode object, or Dual Object. The hyperbolic tangent of each element of x.
    """
    try:
        z = Rnode(np.tanh(x.value))
        x.children.append((1 / np.cosh(x.value)**2, z))
        return z
    except AttributeError:
        try:
            return Dual(np.tanh(x.val), x.der / np.cosh(x.val)**2)
        except AttributeError:
            return np.tanh(x)  # default to numpy implementation


def relu(x: Union[Rnode, Dual, float]) -> Union[Rnode, Dual, float, List[float]]:
    """Calculates the output of the relu function on the input.

    Parameters:
    x : array_like, Rnode object, or Dual Object.

    Returns:
    y : array_like, Rnode object, or Dual Object. The output  of the relu function on each element of x.
    """
    try:

        a = max(0, x.value)
        b = np.where(a > 0, 1, 0)
        z = Rnode(a)
        x.children.append((b, z))
        return z
    except AttributeError:
        try:
            a = max(0, x.val)
            b = np.where(a > 0, 1, 0)
            return Dual(a, b * x.der)
        except AttributeError:
            return max(0, x)


def relu6(x: Union[Rnode, Dual, float]) -> Union[Rnode, Dual, float, List[float]]:
    """Calculates the output of the relu6 function on the input.

    Parameters:
    x : array_like, Rnode object, or Dual Object.

    Returns:
    y : array_like, Rnode object, or Dual Object. The output  of the relu6 function on each element of x.
    """
    try:

        a = float(max(0, x.value))
        b = np.where(0.0 < a < 6.0, 1, 0)
        if a > 6.0:  # clip output to a maximum of 6
            a = 6.0
        z = Rnode(a)
        x.children.append((b, z))
        return z
    except AttributeError:
        try:
            a = float(max(0, x.val))
            b = np.where(0.0 < a < 6.0, 1, 0)
            if a > 6.0:  # clip output to a maximum of 6
                a = 6.0
            return Dual(a, b * x.der)
        except AttributeError:
            return min(max(0, x), 6)


def logistic(x: Union[Rnode, Dual, float]) -> Union[Rnode, Dual, float, List[float]]:
    """Calculates the output of the logistic function given the input.

    Parameters:
    x : array_like, Rnode object, or Dual Object.

    Returns:
    y : array_like, Rnode object, or Dual Object. The output  of the logistic function on each element of x.
    """
    try:
        z = Rnode(1 / (1 + np.exp(-x.value)))
        nominator = np.exp(x.value)
        denominator = (1 + np.exp(x.value)) ** 2
        x.children.append((nominator / denominator, z))
        return z
    except AttributeError:
        try:
            return(1 / (1 + np.exp(-x.val)), np.exp(x.val) / ((1 + np.exp(x.val)) ** 2))
        except AttributeError:
            return 1 / (1 + np.exp(-x))


def exp(x: Union[Rnode, Dual, float]) -> Union[Rnode, Dual, float, List[float]]:
    """Calculates the exponent of input.

    Parameters:
    x : array_like, Rnode object, or Dual Object.

    Returns:
    y : array_like, Rnode object, or Dual Object. The exponent of each element of x.
    """
    try:
        z = Rnode(np.exp(x.value))
        x.children.append((np.exp(x.value), z))
        return z
    except AttributeError:
        try:
            return Dual(np.exp(x.val), np.exp(x.val) * np.asarray(x.der))
        except AttributeError:
            return np.exp(x)  # Default to numpy implementation


# todo: figure out implementation for exp2
# def exp2(x: Union[Rnode, Dual, float]) -> Union[Rnode, Dual, float, List[float]]:
#     """Calculates the double  exponent of input.
#
#     Parameters:
#     x : array_like, Rnode object, or Dual Object.
#
#     Returns:
#     y : array_like, Rnode object, or Dual Object. The exponent of each element of x.
#     """
#     try:
#         return Dual(np.exp2(x._val), np.exp2(x._val) * (x._der * np.log(2)))##, np.log(2) * np.exp2(x._val) * (x._der2 + np.log(2) * x._der**2))
#
#         # return Dual(np.exp2(x._val), np.exp2(x._val) * (x._der * np.log(2)), np.log(2) * np.exp2(x._val) * (x._der2 + np.log(2) * x._der**2))
#     except AttributeError:
#         return np.exp2(x)


def sqrt(x: Union[Rnode, Dual, float]) -> Union[Rnode, Dual, float, List[float]]:
    """Calculates the square root of input.

    Parameters:
    x : array_like, Rnode object, or Dual Object.

    Returns:
    y : array_like, Rnode object, or Dual Object. The square root of each element of x.
    """
    try:
        z = Rnode(x.value ** 0.5)
        # ?
        x.children.append((0.5*x.value ** (-0.5), z))
        return z
    except AttributeError:

        return x.__pow__(0.5)



def power(x: Union[Rnode, Dual, float], power: float) -> Union[Rnode, Dual, float, List[float]]:
    """Calculates the power of input. Alternative method to dunder method __pow__ in Dual class.

    Parameters:
    x : array_like, Rnode object, or Dual Object.
    k : float.

    Returns:
    y : array_like, Rnode object, or Dual Object. Each element of x to the power of 'pow'.
    """
    return x.__pow__(power)


def arcsin(x: Union[Rnode, Dual, float]) -> Union[Rnode, Dual, float, List[float]]:
    """Calculates the inverse sine of the input.

    Parameters:
    x : array_like, Rnode object, or Dual Object.

    Returns:
    y : array_like, Rnode object, or Dual Object. The inverse sine of each element of x.
    """
    try:
        z = Rnode(np.arcsin(x.value))
        temp = 1 - x.value ** 2
        # print("temp is " + str(temp))
        if temp <= 0:
            raise ValueError('Domain of sqrt is {x >= 0}')
        x.children.append((1 / np.sqrt(temp), z))
        return z
    except AttributeError:
        try:
            return Dual(np.arcsin(x.val), 1 / np.sqrt(1 - x.val ** 2) * np.asarray(x.der))
        except AttributeError:
            return np.arcsin(x)


def arccos(x: Union[Rnode, Dual, float]) -> Union[Rnode, Dual, float, List[float]]:
    """Calculates the inverse cosine of the input.

    Parameters:
    x : array_like, Rnode object, or Dual Object.

    Returns:
    y : array_like, Rnode object, or Dual Object. The inverse cosine of each element of x.
    """
    try:
        z = Rnode(np.arccos(x.value))
        temp = 1 - x.value ** 2
        # print("temp is " + str(temp))
        if temp <= 0:
            raise ValueError('Domain of sqrt is {x >= 0}')

        x.children.append((-1 / np.sqrt(temp), z))
        return z
    except AttributeError:
        try:
            return Dual(np.arccos(x.val), -1 / np.sqrt(1 - x.val**2) * np.asarray(x.der))
        except AttributeError:
            return np.arccos(x)


def arctan(x: Union[Rnode, Dual, float]) -> Union[Rnode, Dual, float, List[float]]:
    """Calculates the inverse tangent of the input.

    Parameters:
    x : array_like, Rnode object, or Dual Object.

    Returns:
    y : array_like, Rnode object, or Dual Object. The inverse tangent of each element of x.
    """
    try:
        z = Rnode(np.arctan(x.value))
        x.children.append((1 / (1 + x.value ** 2), z))
        return z
    except AttributeError:
        try:
            return Dual(np.arctan(x.val), 1 / (1 + x.val**2) * np.asarray(x.der))
        except AttributeError:
            return np.arctan(x)


# if __name__ == "__main__":
#     val = Dual(3,[4,1])
#     val2 = Dual(2,[3,1])
#     val + val2
#     z = sin(val)
#     print(z)
#     z = cos(val)
#     print(z)
#     z = tan(val)
#     print(z)
#     print(bool(z))
# if __name__ == "__main__":
#     # define the Vars for the example problem
#     # initialize x = 0.5 and y = 4.2
#     x = Rnode(0.11)
#     y = Rnode(1.0)
#     print('x: ', x)
#     print('y: ', y)
#
#     a = sin(x)
#     a.grad_value = 1.0
#     print("sin partial a/ partial x = {}".format(x.grad())) # ∂a/∂x = 4.2
#     # should print: partial a/ partial x = 0.8775825618903728
#
#     y.clear()
#     b = cos(y)
#     b.grad_value = 1.0
#     print("cos partial b/ partial y = {}".format(y.grad()))  # ∂a/∂x = 4.2
# #?
#     y.clear()
#     b = tan(y)
#     b.grad_value = 1.0
#     print("tan partial b/ partial y = {}".format(y.grad()))  # ∂a/∂x = 4.2
#
#     x.clear()
#     b = arccos(x)
#     b.grad_value = 1.0
#     print("arccos partial b/ partial x = {}".format(x.grad()))  # ∂a/∂x = 4.2
#
#     x.clear()
#     b = log(x)
#     b.grad_value = 1.0
#     print("log partial b/ partial x = {}".format(x.grad()))  # ∂a/∂x = 4.2
#
#     x.clear()
#     b = sinh(x)
#     b.grad_value = 1.0
#     print("sinh partial b/ partial x = {}".format(x.grad()))  # ∂a/∂x = 4.2
#
#     x.clear()
#     b = cosh(x)
#     b.grad_value = 1.0
#     print("cosh partial b/ partial x = {}".format(x.grad()))  # ∂a/∂x = 4.2
#
#     x.clear()
#     b = tanh(x)
#     b.grad_value = 1.0
#     print("tanh partial b/ partial x = {}".format(x.grad()))  # ∂a/∂x = 4.2
#     print('true value is:', 1./np.cosh(x.value))
#
#     x.clear()
#     b = relu(x)
#     b.grad_value = 1.0
#     print("relu partial b/ partial x = {}".format(x.grad()))  # ∂a/∂x = 4.2
#
#     x.clear()
#     b = relu6(x)
#     b.grad_value = 1.0
#     print("relu6 partial b/ partial x = {}".format(x.grad()))
#
#     x.clear()
#     b = logistic(x)
#     b.grad_value = 1.0
#     print("logistic partial b/ partial x = {}".format(x.grad()))  # ∂a/∂x = 4.2
#
#     x.clear()
#     b = exp(x)
#     b.grad_value = 1.0
#     print("exp partial b/ partial x = {}".format(x.grad()))  # ∂a/∂x = 4.2
#
#     x.clear()
#     b = sqrt(x)
#     b.grad_value = 1.0
#     print("sqrt partial b/ partial x = {}".format(x.grad()))  # ∂a/∂x = 4.2
#
#     x.clear()
#     b = arctan(x)
#     b.grad_value = 1.0
#     print("arctan partial b/ partial x = {}".format(x.grad()))  # ∂a/∂x = 4.2
