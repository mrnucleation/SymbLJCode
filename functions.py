"""Functions for use with symbolic regression.

These functions encapsulate multiple implementations (sympy, Tensorflow, numpy) of a particular function so that the
functions can be used in multiple contexts."""

import tensorflow as tf
import numpy as np
import sympy as sp


class BaseFunction:
    """Abstract class for primitive functions"""
    def __init__(self, norm=1):
        self.norm = norm

    def sp(self, x):
        """Sympy implementation"""
        return None

    def tf(self, x):
        """Automatically convert sympy to TensorFlow"""
        z = sp.symbols('z')
        return sp.utilities.lambdify(z, self.sp(z), 'tensorflow')(x)

    def np(self, x):
        """Automatically convert sympy to numpy"""
        z = sp.symbols('z')
        return sp.utilities.lambdify(z, self.sp(z), 'numpy')(x)

    def name(self, x):
        return str(self.sp)


class Constant(BaseFunction):
    def tf(self, x):
        return tf.ones_like(x)

    def sp(self, x):
        return 1

    def np(self, x):
        return np.ones_like


class Identity(BaseFunction):
    def __init__(self, norm=1):
        self.norm = norm
    def tf(self, x):
        return tf.identity(x) / self.norm

    def sp(self, x):
        return x / self.norm

    def np(self, x):
        return np.array(x) / self.norm


class Square(BaseFunction):
    def tf(self, x):
        return tf.square(x) / self.norm

    def sp(self, x):
        return x ** 2 / self.norm

    def np(self, x):
        return np.square(x) / self.norm


class Sqrt(BaseFunction):
    def tf(self, x):
        return tf.where(x < 0.0, tf.zeros_like(x), tf.sqrt(x) / self.norm)

    def sp(self, x):
        return sp.sqrt(x)  / self.norm

    def np(self, x):
        return np.sqrt(x) / self.norm



class Pow(BaseFunction):
    def __init__(self, power, norm=1):
        BaseFunction.__init__(self, norm=norm)
        self.power = power

    def sp(self, x):
        if self.power < 0.0:
            return sp.Piecewise((0, x==0), (x ** self.power / self.norm, True))
        return x ** self.power / self.norm

    def tf(self, x):
        powers = tf.constant(self.power, shape=x.shape)
        if self.power < 0.0:
            x_new = tf.pow(x, powers)
            x_new = tf.where(tf.not_equal(x, 0.), x_new, tf.zeros_like(x))
            return x_new / self.norm
        return tf.pow(x, powers) / self.norm


class Sin(BaseFunction):
    def tf(self, x):
        return tf.sin(x) / self.norm

    def sp(self, x):
        return sp.sin(x) / self.norm

    def np(self, x):
        return np.sin(x) / self.norm

class Cos(BaseFunction):
    def tf(self, x):
        return tf.cos(x) / self.norm

    def sp(self, x):
        return sp.cos(x) / self.norm

    def np(self, x):
        return np.cos(x) / self.norm


class NegExp(BaseFunction):
    def tf(self, x):
        return tf.exp(-x) / self.norm

    def sp(self, x):
        return sp.exp(-x) / self.norm

    def np(self, x):
        return np.exp(-x) / self.norm



class Sigmoid(BaseFunction):
    def tf(self, x):
        return tf.sigmoid(20*x) / self.norm

    def sp(self, x):
        return 1 / (1 + sp.exp(-20*x)) / self.norm

    def np(self, x):
        return 1 / (1 + np.exp(-20*x)) / self.norm

    def name(self, x):
        return "sigmoid(x)"


class Exp(BaseFunction):
    def __init__(self, norm=np.e):
        super().__init__(norm)

    def sp(self, x):
        return (sp.exp(x) - 1) / self.norm



class Log(BaseFunction):
    def sp(self, x):
        return sp.log(sp.Abs(x)) / self.norm


class BaseFunction2:
    """Abstract class for primitive functions with 2 inputs"""
    def __init__(self, norm=1.):
        self.norm = norm

    def sp(self, x, y):
        """Sympy implementation"""
        return None

    def tf(self, x, y):
        """Automatically convert sympy to TensorFlow"""
        a, b = sp.symbols('a b')
        return sp.utilities.lambdify([a, b], self.sp(a, b), 'tensorflow')(x, y)

    def np(self, x, y):
        """Automatically convert sympy to numpy"""
        a, b = sp.symbols('a b')
        return sp.utilities.lambdify([a, b], self.sp(a, b), 'numpy')(x, y)

    def name(self, x, y):
        return str(self.sp)


class Product(BaseFunction2):
    def __init__(self, norm=0.1):
        super().__init__(norm=norm)

    def sp(self, x, y):
        return x*y / self.norm

class PowerXY(BaseFunction2):
    def __init__(self, norm=0.1):
        super().__init__(norm=norm)

    def sp(self, x, y):
        return x**y / self.norm




def count_inputs(funcs):
    i = 0
    for func in funcs:
        if isinstance(func, BaseFunction):
            i += 1
        elif isinstance(func, BaseFunction2):
            i += 2
    return i


def count_double(funcs):
    i = 0
    for func in funcs:
        if isinstance(func, BaseFunction2):
            i += 1
    return i


# default_func = [
#     Constant(),
#     Constant(),
#     Identity(),
#     Identity(),
#     Square(),
#     Square(),
#     Sin(),
#     Sigmoid(),
# ]

default_func = [
    *[Constant()] * 2,
    *[Identity()] * 4,
    *[Square()] * 4,
    *[Sin()] * 2,
    *[Exp()] * 2,
    *[Sigmoid()] * 2,
    *[Product()] * 2,
]
