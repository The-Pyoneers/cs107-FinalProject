from farad.elem import *


class RAutoDiff:
    def __init__(self, fn):
        self.fn = fn
        self.roots = []  # to save the whole tree stucture
        self.value = None
        self.der = []  # more maybe consider a dictionary

    def forwardpass(self,
                    x):  # initialize self.roots, self.value, self.der, #also need to add additional input argument
        # try input a dictionary with names and values
        pass

    def get_val(self):  # return the value of the function
        pass

    def reverse(self, varname):  # return the derivative with respect to varname variable
        pass

if __name__ == "__main__":
    #test for driver

    def fi(x):
        return x+sin(x)*exp(cos(x))

    autofx = RAutoDiff(fi)
    autofx.forwardpass(x=4)
    print('value of fi is:', autofx.get_val())
    print('value of fi is:', autofx.reverse(''))


    def fi2d(x, y):
        return x+sin(x) + x * y

    autofx2d = RAutoDiff(fi2d)
    autofx2d.forwardpass(x=4, y=4)
    print('value of fi is:', autofx2d.get_val())
    print('value of (partial fi/ partial x) is:', autofx2d.reverse('x'))
    print('value of (partial fi/ partial y) is:', autofx2d.reverse('y'))