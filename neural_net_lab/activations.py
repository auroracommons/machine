"""
Activation functions for neural network layers.

Each activation is implemented as a class with forward() and backward() methods
that operate on numpy arrays. The forward pass caches values needed for the
backward pass.
"""

import numpy as np


class Activation:
    """Base class for activation functions."""

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)


class ReLU(Activation):
    """Rectified Linear Unit: max(0, x)."""

    def forward(self, x):
        self._mask = (x > 0).astype(np.float64)
        return x * self._mask

    def backward(self, grad_output):
        return grad_output * self._mask


class LeakyReLU(Activation):
    """Leaky ReLU: x if x > 0, else alpha * x."""

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self._mask = (x > 0).astype(np.float64)
        self._x = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, grad_output):
        dx = np.where(self._x > 0, 1.0, self.alpha)
        return grad_output * dx


class Sigmoid(Activation):
    """Sigmoid: 1 / (1 + exp(-x))."""

    def forward(self, x):
        # Numerically stable sigmoid
        self._out = np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x))
        )
        return self._out

    def backward(self, grad_output):
        return grad_output * self._out * (1.0 - self._out)


class Tanh(Activation):
    """Hyperbolic tangent activation."""

    def forward(self, x):
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad_output):
        return grad_output * (1.0 - self._out ** 2)


class Softmax(Activation):
    """
    Softmax activation: exp(x_i) / sum(exp(x_j)).
    Applied along the last axis (per-sample for batched input).
    """

    def forward(self, x):
        # Numerical stability: subtract max
        shifted = x - x.max(axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        self._out = exp_x / exp_x.sum(axis=-1, keepdims=True)
        return self._out

    def backward(self, grad_output):
        # For each sample: dL/dx_i = s_i * (dL/ds_i - sum_j(dL/ds_j * s_j))
        s = self._out
        dot = np.sum(grad_output * s, axis=-1, keepdims=True)
        return s * (grad_output - dot)


class GELU(Activation):
    """
    Gaussian Error Linear Unit.
    GELU(x) = x * Phi(x) where Phi is the CDF of standard normal.
    Uses the tanh approximation for efficiency.
    """

    def forward(self, x):
        self._x = x
        # Tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        self._inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
        self._tanh = np.tanh(self._inner)
        return 0.5 * x * (1.0 + self._tanh)

    def backward(self, grad_output):
        x = self._x
        sech2 = 1.0 - self._tanh ** 2
        inner_deriv = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * x ** 2)
        dx = 0.5 * (1.0 + self._tanh) + 0.5 * x * sech2 * inner_deriv
        return grad_output * dx


# Registry for easy lookup by name
ACTIVATIONS = {
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax,
    "gelu": GELU,
}


def get_activation(name):
    """Get an activation function by name."""
    if name is None:
        return None
    if isinstance(name, Activation):
        return name
    name = name.lower()
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. Available: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]()
