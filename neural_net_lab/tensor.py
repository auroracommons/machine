"""
Custom Tensor class with automatic differentiation support.

Implements a computational graph that tracks operations and enables
reverse-mode automatic differentiation (backpropagation).
"""

import numpy as np


class Tensor:
    """
    A tensor that tracks its computational history for automatic differentiation.

    Each Tensor wraps a numpy array and optionally records the operation that
    created it and its parent tensors. During the backward pass, gradients
    flow back through the graph using the chain rule.
    """

    def __init__(self, data, requires_grad=False, _children=(), _op="", label=""):
        if isinstance(data, (int, float)):
            data = np.array(data, dtype=np.float64)
        elif isinstance(data, list):
            data = np.array(data, dtype=np.float64)
        elif isinstance(data, np.ndarray):
            data = data.astype(np.float64) if data.dtype != np.float64 else data
        else:
            data = np.array(data, dtype=np.float64)

        self.data = data
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.shape = self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    # ------------------------------------------------------------------ #
    #  Arithmetic operations (build computational graph)
    # ------------------------------------------------------------------ #

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=True, _children=(self, other), _op="+")

        def _backward():
            # Sum out broadcast dimensions
            self_grad = out.grad
            other_grad = out.grad

            # Handle broadcasting for self
            if self.data.shape != out.data.shape:
                # Sum over axes that were broadcast
                ndim_diff = len(out.data.shape) - len(self.data.shape)
                axes = list(range(ndim_diff))
                for i, (s, o) in enumerate(zip(self.data.shape, out.data.shape[ndim_diff:])):
                    if s == 1 and o != 1:
                        axes.append(i + ndim_diff)
                self_grad = self_grad.sum(axis=tuple(axes), keepdims=True)
                if ndim_diff > 0:
                    self_grad = self_grad.reshape(self.data.shape)

            # Handle broadcasting for other
            if other.data.shape != out.data.shape:
                ndim_diff = len(out.data.shape) - len(other.data.shape)
                axes = list(range(ndim_diff))
                for i, (s, o) in enumerate(zip(other.data.shape, out.data.shape[ndim_diff:])):
                    if s == 1 and o != 1:
                        axes.append(i + ndim_diff)
                other_grad = other_grad.sum(axis=tuple(axes), keepdims=True)
                if ndim_diff > 0:
                    other_grad = other_grad.reshape(other.data.shape)

            self.grad = self.grad + self_grad
            other.grad = other.grad + other_grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=True, _children=(self,), _op="neg")

        def _backward():
            self.grad = self.grad + (-out.grad)

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=True, _children=(self, other), _op="*")

        def _backward():
            self_grad = other.data * out.grad
            other_grad = self.data * out.grad

            if self.data.shape != out.data.shape:
                ndim_diff = len(out.data.shape) - len(self.data.shape)
                axes = list(range(ndim_diff))
                for i, (s, o) in enumerate(zip(self.data.shape, out.data.shape[ndim_diff:])):
                    if s == 1 and o != 1:
                        axes.append(i + ndim_diff)
                self_grad = self_grad.sum(axis=tuple(axes), keepdims=True)
                if ndim_diff > 0:
                    self_grad = self_grad.reshape(self.data.shape)

            if other.data.shape != out.data.shape:
                ndim_diff = len(out.data.shape) - len(other.data.shape)
                axes = list(range(ndim_diff))
                for i, (s, o) in enumerate(zip(other.data.shape, out.data.shape[ndim_diff:])):
                    if s == 1 and o != 1:
                        axes.append(i + ndim_diff)
                other_grad = other_grad.sum(axis=tuple(axes), keepdims=True)
                if ndim_diff > 0:
                    other_grad = other_grad.reshape(other.data.shape)

            self.grad = self.grad + self_grad
            other.grad = other.grad + other_grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other ** (-1.0)

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other * self ** (-1.0)

    def __pow__(self, exponent):
        if isinstance(exponent, Tensor):
            exponent_val = exponent.data
        else:
            exponent_val = exponent
        out = Tensor(self.data ** exponent_val, requires_grad=True, _children=(self,), _op=f"**{exponent_val}")

        def _backward():
            self.grad = self.grad + exponent_val * (self.data ** (exponent_val - 1)) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    #  Matrix operations
    # ------------------------------------------------------------------ #

    def matmul(self, other):
        """Matrix multiplication: self @ other."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=True, _children=(self, other), _op="@")

        def _backward():
            # For 2D: dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC
            if self.data.ndim == 2 and other.data.ndim == 2:
                self.grad = self.grad + out.grad @ other.data.T
                other.grad = other.grad + self.data.T @ out.grad
            elif self.data.ndim == 1 and other.data.ndim == 2:
                # (N,) @ (N, M) -> (M,)
                self.grad = self.grad + out.grad @ other.data.T
                other.grad = other.grad + self.data.reshape(-1, 1) @ out.grad.reshape(1, -1)
            else:
                # General case
                self.grad = self.grad + out.grad @ np.swapaxes(other.data, -2, -1)
                other.grad = other.grad + np.swapaxes(self.data, -2, -1) @ out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    # ------------------------------------------------------------------ #
    #  Reduction operations
    # ------------------------------------------------------------------ #

    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=True, _children=(self,), _op="sum")

        def _backward():
            if axis is None:
                self.grad = self.grad + np.ones_like(self.data) * out.grad
            else:
                grad = out.grad
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad = self.grad + np.ones_like(self.data) * grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out_data = self.data.mean(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=True, _children=(self,), _op="mean")

        def _backward():
            if axis is None:
                n = self.data.size
                self.grad = self.grad + np.ones_like(self.data) * out.grad / n
            else:
                grad = out.grad
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                if isinstance(axis, int):
                    n = self.data.shape[axis]
                else:
                    n = np.prod([self.data.shape[a] for a in axis])
                self.grad = self.grad + np.ones_like(self.data) * grad / n

        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        out_data = self.data.max(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=True, _children=(self,), _op="max")

        def _backward():
            if axis is None:
                mask = (self.data == out_data).astype(np.float64)
                mask = mask / mask.sum()  # distribute gradient evenly to ties
                self.grad = self.grad + mask * out.grad
            else:
                grad = out.grad
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                    expanded_out = np.expand_dims(out_data, axis=axis)
                else:
                    expanded_out = out_data
                mask = (self.data == expanded_out).astype(np.float64)
                counts = mask.sum(axis=axis, keepdims=True)
                counts = np.maximum(counts, 1)
                mask = mask / counts
                self.grad = self.grad + mask * grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    #  Unary operations
    # ------------------------------------------------------------------ #

    def exp(self):
        out = Tensor(np.exp(np.clip(self.data, -500, 500)), requires_grad=True, _children=(self,), _op="exp")

        def _backward():
            self.grad = self.grad + out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(np.clip(self.data, 1e-12, None)), requires_grad=True, _children=(self,), _op="log")

        def _backward():
            self.grad = self.grad + out.grad / np.clip(self.data, 1e-12, None)

        out._backward = _backward
        return out

    def sqrt(self):
        return self ** 0.5

    def abs(self):
        out = Tensor(np.abs(self.data), requires_grad=True, _children=(self,), _op="abs")

        def _backward():
            self.grad = self.grad + np.sign(self.data) * out.grad

        out._backward = _backward
        return out

    def clip(self, min_val, max_val):
        out = Tensor(np.clip(self.data, min_val, max_val), requires_grad=True, _children=(self,), _op="clip")

        def _backward():
            mask = ((self.data >= min_val) & (self.data <= max_val)).astype(np.float64)
            self.grad = self.grad + mask * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    #  Shape operations
    # ------------------------------------------------------------------ #

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), requires_grad=True, _children=(self,), _op="reshape")

        def _backward():
            self.grad = self.grad + out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        out = Tensor(self.data.transpose(axes), requires_grad=True, _children=(self,), _op="T")

        def _backward():
            if axes is None:
                self.grad = self.grad + out.grad.transpose()
            else:
                # Invert the permutation
                inv = [0] * len(axes)
                for i, a in enumerate(axes):
                    inv[a] = i
                self.grad = self.grad + out.grad.transpose(inv)

        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    # ------------------------------------------------------------------ #
    #  Backward pass
    # ------------------------------------------------------------------ #

    def backward(self, gradient=None):
        """Reverse-mode automatic differentiation."""
        if gradient is None:
            if self.data.size == 1:
                self.grad = np.ones_like(self.data)
            else:
                raise ValueError("Gradient must be specified for non-scalar tensors")
        else:
            if isinstance(gradient, Tensor):
                self.grad = gradient.data.copy()
            else:
                self.grad = np.array(gradient, dtype=np.float64)

        # Topological sort
        topo = []
        visited = set()

        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Backprop in reverse topological order
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        """Reset gradient to zeros."""
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #

    def detach(self):
        """Return a new tensor with no computational history."""
        return Tensor(self.data.copy(), requires_grad=False)

    def numpy(self):
        """Return the underlying numpy array."""
        return self.data

    def item(self):
        """Return scalar value."""
        return self.data.item()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], requires_grad=True, _children=(self,), _op="getitem")

        def _backward():
            full_grad = np.zeros_like(self.data)
            full_grad[idx] = out.grad
            self.grad = self.grad + full_grad

        out._backward = _backward
        return out
