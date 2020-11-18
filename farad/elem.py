"""Math functionality for farad package."""

__all__ = ['sin', 'cos', 'tan', 'log', 'log10', 'sinh', 'cosh', 'tanh' \
           'log2', 'exp', 'exp2', 'sqrt', 'arccos', 'arcsin', 'arctan' \
           'relu', 'logistic', 'relu6']


from farad.dual import Dual
import numpy as np
from typing import Union, List


def sin(x: Dual) -> Union[Dual, float, List[float]]:
    """Calculate sine of the input in radians.

    Parameters:	
    x : array_like or Dual Object. Angle, in radians (2 pi rad equals 360 degrees).

    Returns:	
    y : array_like or Dual Object. The sine of each element of x.
    """
    try:  # Python EAFP principle - assume input is type Dual
        return Dual(np.sin(x._val), np.cos(x._val)*x._der)
    except AttributeError:
        return np.sin(x)  # Default to numpy implementation


def cos(x: Union[Dual, float]) -> Union[Dual, float, List[float]]:
    """Calculate cosine of the input in radians.

    Parameters:	
    x : array_like or Dual Object. Angle, in radians (2 pi rad equals 360 degrees).

    Returns:	
    y : array_like or Dual Object. The cosine of each element of x.
    """
    try:
        return Dual(np.cos(x._val), -np.sin(x._val)*x._der)
    except AttributeError:
        return np.cos(x)  # Default to numpy implementation


def tan(x: Union[Dual, float]) -> Union[Dual, float, List[float]]:
    """Calculate tangent of the input in radians.

    Parameters:	
    x : array_like or Dual Object. Angle, in radians (2 pi rad equals 360 degrees).

    Returns:	
    y : array_like or Dual Object. The tangent of each element of x.
    """
    try:
        return Dual(np.tan(x._val), 1/np.cos(x._val)**2*x._der)
    except AttributeError:
        return np.tan(x)  # Default to numpy implementation


def log(x: Union[Dual, float]) -> Union[Dual, float, List[float]]:
    """Calculate natural logarithm of input.

    Parameters:	
    x : array_like or Dual Object. 

    Returns:	
    y : array_like or Dual Object. The natural logarithm of each element of x.
    """
    try:
        if x._val <= 0:
            raise ValueError('Domain of logarithm is {x > 0}')
        return Dual(np.log(x._val), (1/x._val)*x._der)
    except AttributeError:
        return np.log(x)  # Default to numpy implementation


def log10(x: Union[Dual, float]) -> Union[Dual, float, List[float]]:
    """Calculates the base-10 logarithm of input.

    Parameters:	
    x : array_like or Dual Object. 

    Returns:	
    y : array_like or Dual Object. The base-10 logarithm of each element of x.
    """
    try:
        if x._val <= 0:
            raise ValueError('Domain of logarithm is {x > 0}')
        return Dual(np.log10(x._val), (1/(x._val*np.log(10)))*x._der)
    except AttributeError:
        return np.log10(x)  # Default to numpy implementation


def log2(x: Union[Dual, float]) -> Union[Dual, float, List[float]]:
    """Calculates the base-2 logarithm of input.

    Parameters:	
    x : array_like or Dual Object. 

    Returns:	
    y : array_like or Dual Object. The base-2 logarithm of each element of x.
    """
    try:
        if x._val <= 0:
            raise ValueError('Domain of logarithm is {x > 0}')
        return Dual(np.log2(x._val), (1/(x._val*np.log(2)))*x._der)
    except AttributeError:
        return np.log2(x)  # Default to numpy implementation


def sinh(x: Union[Dual, float]) -> Union[Dual, float, List[float]]:
    """Calculates the hyberbolic sine of input.

    Parameters:	
    x : array_like or Dual Object. 

    Returns:	
    y : array_like or Dual Object. The hyperbolic sine of each element of x.
    """
    try:
        return Dual(np.sinh(x.val), np.cosh(x.val) * x.der)
    except AttributeError:
        return np.sinh(x)  # default to numpy implementation


def cosh(x: Union[Dual, float]) -> Union[Dual, float, List[float]]:
    """Calculates the hyberbolic cosine of input.

    Parameters:	
    x : array_like or Dual Object. 

    Returns:	
    y : array_like or Dual Object. The hyperbolic cosine of each element of x.
    """
    try:
        return Dual(np.cosh(x.val), np.sinh(x.val) * x.der)
    except AttributeError:
        return np.cosh(x)  # default to numpy implementation


def tanh(x: Union[Dual, float]) -> Union[Dual, float, List[float]]:
    """Calculates the hyberbolic tangent of input.

    Parameters:	
    x : array_like or Dual Object. 

    Returns:	
    y : array_like or Dual Object. The hyperbolic tangent of each element of x.
    """
    try:
        return Dual(np.tanh(x.val), x.der / np.cosh(x.val))
    except AttributeError:
        return np.tanh(x)  # default to numpy implementation


def relu(x: Union[Dual, float]) -> Union[Dual, float, List[float]]:
    """Calculates the output of the relu function on the input.

    Parameters:	
    x : array_like or Dual Object. 

    Returns:	
    y : array_like or Dual Object. The output  of the relu function on each element of x.
    """
    try:
        a = max(0, x.val)
        b = np.where(a > 0, 1, 0)
        return Dual(a, b * x.der)
    except AttributeError:
        return max(0, x)


def relu6(x: Union[Dual, float]) -> Union[Dual, float, List[float]]:
    """Calculates the output of the relu6 function on the input.

    Parameters:	
    x : array_like or Dual Object. 

    Returns:	
    y : array_like or Dual Object. The output  of the relu6 function on each element of x.
    """
    try:
        a = max(0, x.val)
        b = np.where(a > 0, 1, 0)
        if a > 6:  # clip output to a maximum of 6
            a = 6
        return Dual(a, b * x.der)
    except AttributeError:
        return min(max(0, x), 6)


def logistic(x: Union[Dual, float]) -> Union[Dual, float, List[float]]:
    """Calculates the output of the logistic function given the input.

    Parameters:	
    x : array_like or Dual Object. 

    Returns:	
    y : array_like or Dual Object. The output  of the logistic function on each element of x.
    """
    try:
        return(1 / (1 + np.exp(-x.val)), np.exp(x.val) / ((1 + np.exp(x.val)) ** 2))
    except AttributeError:
       return 1 / (1 + np.exp(-x))


def exp(x: Union[Dual, float]) -> Union[Dual, float, List[float]]:
    """Calculates the exponent of input.

    Parameters:	
    x : array_like or Dual Object. 

    Returns:	
    y : array_like or Dual Object. The exponent of each element of x.
    """
    try:
        return Dual(np.exp(x.val), np.exp(x.val) * x.der)
    except AttributeError:
        return np.exp(x)  # Default to numpy implementation


# todo: figure out implementation for exp2
# def exp2(x: Union[Dual, float]) -> Union[Dual, float, List[float]]:
#     """Calculates the double  exponent of input.
#
#     Parameters:
#     x : array_like or Dual Object.
#
#     Returns:
#     y : array_like or Dual Object. The exponent of each element of x.
#     """
#     try:
#         return Dual(np.exp2(x._val), np.exp2(x._val) * (x._der * np.log(2)))##, np.log(2) * np.exp2(x._val) * (x._der2 + np.log(2) * x._der**2))
#
#         # return Dual(np.exp2(x._val), np.exp2(x._val) * (x._der * np.log(2)), np.log(2) * np.exp2(x._val) * (x._der2 + np.log(2) * x._der**2))
#     except AttributeError:
#         return np.exp2(x)


def sqrt(x: Union[Dual, float]) -> Union[Dual, float, List[float]]:
    """Calculates the square root of input.

    Parameters:	
    x : array_like or Dual Object. 

    Returns:	
    y : array_like or Dual Object. The square root of each element of x.
    """
    try:
        return x.__pow__(0.5)
    except AttributeError:
        return np.sqrt(x)  # default to numpy implementation


def power(x: Union[Dual, float], power: float) -> Union[Dual, float, List[float]]:
    """Calculates the power of input. Alternative method to dunder method __pow__ in Dual class.

    Parameters:	
    x : array_like or Dual Object. 
    k : float.

    Returns:	
    y : array_like or Dual Object. Each element of x to the power of 'pow'.
    """
    try:
        return x.__pow__(power)
    except AttributeError:
        return np.power(x, power)  # default to numpy implementation


def arcsin(x: Union[Dual, float]) -> Union[Dual, float, List[float]]:
    """Calculates the inverse sine of the input.

    Parameters:	
    x : array_like or Dual Object. 

    Returns:	
    y : array_like or Dual Object. The inverse sine of each element of x.
    """
    try:
        return Dual(np.arcsin(x.val), 1 / np.sqrt(1 - x.val **2) * x.der)
    except AttributeError:
        return np.arcsin(x)


def arccos(x: Union[Dual, float]) -> Union[Dual, float, List[float]]:
    """Calculates the inverse cosine of the input.

    Parameters:	
    x : array_like or Dual Object. 

    Returns:	
    y : array_like or Dual Object. The inverse cosine of each element of x.
    """
    try:
        return Dual(np.arccos(x.val), -1 / np.sqrt(1 - x.val**2) * x.der)
    except AttributeError:
        return np.arccos(x)


def arctan(x: Union[Dual, float]) -> Union[Dual, float, List[float]]:
    """Calculates the inverse tangent of the input.

    Parameters:	
    x : array_like or Dual Object. 

    Returns:	
    y : array_like or Dual Object. The inverse tangent of each element of x.
    """
    try:
        return Dual(np.arctan(x.val), 1 / (1 + x.val**2) * x.der)
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