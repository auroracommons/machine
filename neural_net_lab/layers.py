"""
Neural network layers.

Each layer maintains its own parameters and implements forward/backward methods.
Parameters are stored as numpy arrays with accompanying gradient arrays.
"""

import numpy as np
from .activations import get_activation


class Parameter:
    """A trainable parameter with its gradient."""

    def __init__(self, data):
        self.data = data.astype(np.float64)
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    def zero_grad(self):
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    @property
    def shape(self):
        return self.data.shape


class Layer:
    """Base class for all layers."""

    def __init__(self):
        self._training = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        """Return list of trainable Parameters."""
        return []

    def train(self):
        self._training = True

    def eval(self):
        self._training = False


class Dense(Layer):
    """
    Fully connected (dense) layer.

    Computes: output = input @ W + b

    Weight initialization strategies:
      - 'xavier': Xavier/Glorot initialization (good for sigmoid/tanh)
      - 'he': He initialization (good for ReLU)
      - 'normal': Standard normal scaled by 0.01
      - 'zeros': All zeros (not recommended)
    """

    def __init__(self, in_features, out_features, activation=None,
                 weight_init="he", use_bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.activation = get_activation(activation)

        # Initialize weights
        if weight_init == "xavier":
            # Xavier / Glorot: Var = 2 / (fan_in + fan_out)
            std = np.sqrt(2.0 / (in_features + out_features))
            w = np.random.randn(in_features, out_features) * std
        elif weight_init == "he":
            # He / Kaiming: Var = 2 / fan_in
            std = np.sqrt(2.0 / in_features)
            w = np.random.randn(in_features, out_features) * std
        elif weight_init == "normal":
            w = np.random.randn(in_features, out_features) * 0.01
        elif weight_init == "zeros":
            w = np.zeros((in_features, out_features))
        else:
            raise ValueError(f"Unknown weight init: {weight_init}")

        self.weight = Parameter(w)
        if use_bias:
            self.bias = Parameter(np.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        self._input = x
        out = x @ self.weight.data
        if self.use_bias:
            out = out + self.bias.data
        if self.activation is not None:
            out = self.activation.forward(out)
        return out

    def backward(self, grad_output):
        if self.activation is not None:
            grad_output = self.activation.backward(grad_output)

        # Gradient w.r.t. weights: X^T @ grad_output
        self.weight.grad += self._input.T @ grad_output

        # Gradient w.r.t. bias: sum over batch dimension
        if self.use_bias:
            self.bias.grad += grad_output.sum(axis=0)

        # Gradient w.r.t. input: grad_output @ W^T
        grad_input = grad_output @ self.weight.data.T
        return grad_input

    def parameters(self):
        params = [self.weight]
        if self.use_bias:
            params.append(self.bias)
        return params


class Dropout(Layer):
    """
    Dropout regularization layer.

    During training, randomly zeroes elements with probability p,
    and scales the remaining elements by 1/(1-p) (inverted dropout).
    During evaluation, acts as identity.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self._training and self.p > 0:
            self._mask = (np.random.rand(*x.shape) > self.p).astype(np.float64)
            self._mask /= (1.0 - self.p)  # Inverted dropout scaling
            return x * self._mask
        return x

    def backward(self, grad_output):
        if self._training and self.p > 0:
            return grad_output * self._mask
        return grad_output


class BatchNorm(Layer):
    """
    Batch Normalization layer.

    Normalizes each feature across the batch to have zero mean and unit variance,
    then applies a learnable affine transform: y = gamma * x_hat + beta.

    During training, uses batch statistics. During evaluation, uses running
    averages accumulated during training.
    """

    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # Learnable parameters
        self.gamma = Parameter(np.ones(num_features))
        self.beta = Parameter(np.zeros(num_features))

        # Running statistics for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x):
        if self._training:
            # Compute batch statistics
            self._batch_mean = x.mean(axis=0)
            self._batch_var = x.var(axis=0)

            # Normalize
            self._x_hat = (x - self._batch_mean) / np.sqrt(self._batch_var + self.eps)
            self._x = x
            self._std_inv = 1.0 / np.sqrt(self._batch_var + self.eps)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self._batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self._batch_var

            return self.gamma.data * self._x_hat + self.beta.data
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma.data * x_hat + self.beta.data

    def backward(self, grad_output):
        N = grad_output.shape[0]

        # Gradients of learnable parameters
        self.gamma.grad += (grad_output * self._x_hat).sum(axis=0)
        self.beta.grad += grad_output.sum(axis=0)

        # Gradient w.r.t. input (using the batch norm backward formulas)
        dx_hat = grad_output * self.gamma.data
        dvar = np.sum(dx_hat * (self._x - self._batch_mean) * (-0.5) * (self._std_inv ** 3), axis=0)
        dmean = np.sum(dx_hat * (-self._std_inv), axis=0) + dvar * np.mean(-2.0 * (self._x - self._batch_mean), axis=0)

        dx = dx_hat * self._std_inv + dvar * 2.0 * (self._x - self._batch_mean) / N + dmean / N
        return dx

    def parameters(self):
        return [self.gamma, self.beta]
