"""Math functionality for farad package.
"""

__all__ = ['sin', 'cos', 'tan', 'log', 'log10', 'relu', 'sinh', 'cosh', 'tanh' \
           'sigmoid', 'log2', 'exp', 'exp2', 'sqrt']

from dual import Dual
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


def cos(x: Dual) -> Union[Dual, float, List[float]]:
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


def tan(x: Dual) -> Union[Dual, float, List[float]]:
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


def log(x: Dual) -> Union[Dual, float, List[float]]:
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


def log10(x: Dual) -> Union[Dual, float, List[float]]:
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


def log2(x: Dual) -> Union[Dual, float, List[float]]:
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


def sinh(x: Dual) -> Union[Dual, float, List[float]]:
    pass


def cosh(x: Dual) -> Union[Dual, float, List[float]]:
    pass


def tanh(x: Dual) -> Union[Dual, float, List[float]]:
    pass


def relu(x: Dual) -> Union[Dual, float, List[float]]:
    pass


def sigmoid(x: Dual) -> Union[Dual, float, List[float]]:
    pass


def exp(x: Dual) -> Union[Dual, float, List[float]]:
    """Calculates the exponent of input.

    Parameters:	
    x : array_like or Dual Object. 

    Returns:	
    y : array_like or Dual Object. The exponent of each element of x.
    """
    try:
        return Dual(np.exp(x.val), np.exp(x.val) * x.der)
    except AttributeError:
        return np.exp()  # Default to numpy implementation

def exp2(x: Dual) -> Union[Dual, float, List[float]]:
    pass


def sqrt(x: Dual) -> Union[Dual, float, List[float]]:
    pass


if __name__ == "__main__":
    val = Dual(3,[4,1])
    val2 = Dual(2,[3,1])
    val + val2
    z = sin(val)
    print(z)
    z = cos(val)
    print(z)
    z = tan(val)
    print(z)
    print(bool(z))