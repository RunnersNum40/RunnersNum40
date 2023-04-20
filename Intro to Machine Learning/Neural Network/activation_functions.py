import math
import numpy as np

class Activation:
    def activate(self, x: list) -> list:
        raise NotImplementedError

    def derivative(self, x: list) -> list:
        raise NotImplementedError


class Identity(Activation):
    def activate(self, x: list) -> list:
        return x

    def derivative(self, x: list) -> list:
        return np.full_like(x, 1)

class Sigmoid(Activation):
    def activate(self, x: list) -> list:
        return 1/(1+np.exp(-x))

    def derivative(self, x: list) -> list:
        sig = self.activate(x)
        return sig*(1-sig)

class Tanh(Activation):
    def activate(self, x: list) -> list:
        return np.tanh(x)

    def derivative(self, x: list) -> list:
        return 1-np.square(np.tanh(x))

class Lelu(Activation):
    def __init__(self, a: float = 0.1):
        super().__init__()
        self.a = a

    def activate(self, x: list) -> list:
        return (x > 0)*x+(x <= 0)*self.a*x

    def derivative(self, x: list) -> list:
        return (x > 0)+(x <= 0)*self.a

class Relu(Lelu):
    def __init__(self):
        super().__init__(0)

class Step(Activation):
    def activate(self, x: list) -> list:
        return np.array(x >= 0, dtype=np.int32)

    def derivative(self, x: list) -> list:
        return np.full_like(x, 0)

if __name__ == '__main__':
    test_input = np.random.rand(5)
    functions = [Identity(), Sigmoid(), Tanh(), Lelu(), Relu(), Step()]

    print(test_input)
    for function in functions:
        print()
        print(function.activate(test_input))
        print(function.derivative(test_input))
