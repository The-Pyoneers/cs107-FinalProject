# a demo showing the application of farad package in root-finding algorithm by
# Newton's method. For now, we simply consider the case of scalar function with
# scalar input.


from farad.dual import Dual
import farad.elem as el
import numpy as np

# an example input function
def fx(x):
    return np.sin(x) - x


# the derivative function of fx
# users need to rewrite the fx in using elementary
# functions overloaded in farad package.
def dfx(x):
    xd = Dual(x, 1)
    fx = el.sin(xd) - xd
    return fx.der


def newton_root(f, df, x0, epsilon, max_iter):
    '''
    an algorithm to find the root of f(x)=0 by Newton's method

    Parameters:
    f: function. Input function f(x)
    df: function. the derivative of input function: df(x) = df(x)/dx
    x0: number. the initial guess of the root value
    epsilon: number. The accuracy threshold. We claim x to be the root when
                     f(x) < epsilon
    max_iter: number. the maximum iterations the routine can take before stop

    Returns:
    y: number or None. When the routine successfully finds the root, the value
    of the root will be returned. Otherwise, None will be returned.
    '''
    n = 1
    xn = x0
    while n <= max_iter:
        fx = f(xn)
        dfx = df(xn)
        if abs(fx) < epsilon:
            print("Successfully found the root after", n, "iterations")
            return xn
        if dfx == 0:
            print("Failed to find the root because derivative is 0.")
            return None
        xn = xn - fx/dfx
        n = n + 1
    print("Reached maximum iterations. Failed to find the root")
    return None


if __name__ == "__main__":
    root = newton_root(fx, dfx, 0.1, 1.e-20, 2000)
    print("root from root-finding function is ", root)
    print('the accurate root is 0.')
