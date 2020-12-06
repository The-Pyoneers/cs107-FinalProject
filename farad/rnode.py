import numpy as np


class Rnode:

    def __init__(self, value):
        self.value = value
        self.children = []
        self.grad_value = None

    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        pass

    def __add__(self, other):
        try:
            z = Rnode(self.value + other.value)
            self.children.append((1., z)) # weight = ∂z/∂self = other.value
            other.children.append((1., z)) # weight = ∂z/∂other = self.value
        except AttributeError:
            z = Rnode(self.value + other)
            self.children.append((1., z))
        return z

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        try:
            z = Rnode(self.value - other.value)
            self.children.append((1., z))
            other.children.append((-1., z))
        except AttributeError:
            z = Rnode(self.value - other)
            self.children.append((1., z))
        return z

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        try:
            z = Rnode(self.value * other.value)
            self.children.append((other.value, z))
            other.children.append((self.value, z))
        except AttributeError:
            z = Rnode(self.value * other)
            self.children.append((other, z))
        return z

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        try:
            z = Rnode(self.value ** other.value)
            self.children.append((other.value* self.value ** (other.value - 1.), z))
            other.children.append((self.value ** other.value * np.log(self.value), z))
        except AttributeError:
            z = Rnode(self.value ** other)
            self.children.append((other* self.value ** (other - 1.), z))
        return z

    def __rpow__(self, other):
        pass

    def __truediv__(self, other):
        pass

    def __rtruediv__(self, other):
        pass

    def grad(self):
        # recurse only if the value is not yet cached
        if self.grad_value is None:
            # calculate derivative using chain rule
            self.grad_value = sum(weight * var.grad()
                                  for weight, var in self.children)
        return self.grad_value


