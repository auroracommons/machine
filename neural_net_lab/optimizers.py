"""
Optimizers for neural network training.

Each optimizer updates model parameters based on their gradients.
All optimizers support gradient clipping.
"""

import numpy as np


class Optimizer:
    """Base optimizer class."""

    def __init__(self, parameters, lr=0.01, grad_clip=None):
        self.parameters = list(parameters)
        self.lr = lr
        self.grad_clip = grad_clip
        self.t = 0  # step counter

    def zero_grad(self):
        """Reset all parameter gradients to zero."""
        for param in self.parameters:
            param.zero_grad()

    def _clip_gradients(self):
        """Clip gradients by global norm if grad_clip is set."""
        if self.grad_clip is not None:
            total_norm = 0.0
            for param in self.parameters:
                total_norm += np.sum(param.grad ** 2)
            total_norm = np.sqrt(total_norm)

            if total_norm > self.grad_clip:
                scale = self.grad_clip / (total_norm + 1e-8)
                for param in self.parameters:
                    param.grad *= scale

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum and weight decay.

    With momentum:
      v_t = momentum * v_{t-1} + grad
      param -= lr * v_t

    With Nesterov momentum:
      v_t = momentum * v_{t-1} + grad
      param -= lr * (momentum * v_t + grad)
    """

    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0,
                 nesterov=False, grad_clip=None):
        super().__init__(parameters, lr, grad_clip)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self._velocities = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        self._clip_gradients()
        self.t += 1

        for i, param in enumerate(self.parameters):
            grad = param.grad.copy()

            # Weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data

            if self.momentum != 0:
                self._velocities[i] = self.momentum * self._velocities[i] + grad
                if self.nesterov:
                    update = self.momentum * self._velocities[i] + grad
                else:
                    update = self._velocities[i]
            else:
                update = grad

            param.data -= self.lr * update


class Adam(Optimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).

    Maintains per-parameter first moment (mean) and second moment (variance)
    estimates with bias correction.

    m_t = beta1 * m_{t-1} + (1-beta1) * grad
    v_t = beta2 * v_{t-1} + (1-beta2) * grad^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    param -= lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0, grad_clip=None):
        super().__init__(parameters, lr, grad_clip)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [np.zeros_like(p.data) for p in self.parameters]
        self._v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        self._clip_gradients()
        self.t += 1

        for i, param in enumerate(self.parameters):
            grad = param.grad.copy()

            if self.weight_decay != 0:
                grad += self.weight_decay * param.data

            # Update biased first and second moment estimates
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * grad
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = self._m[i] / (1 - self.beta1 ** self.t)
            v_hat = self._v[i] / (1 - self.beta2 ** self.t)

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class RMSProp(Optimizer):
    """
    RMSProp optimizer.

    Divides the learning rate by a running average of the magnitudes of
    recent gradients.

    v_t = rho * v_{t-1} + (1-rho) * grad^2
    param -= lr * grad / (sqrt(v_t) + eps)
    """

    def __init__(self, parameters, lr=0.001, rho=0.9, eps=1e-8,
                 weight_decay=0.0, grad_clip=None):
        super().__init__(parameters, lr, grad_clip)
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        self._v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        self._clip_gradients()
        self.t += 1

        for i, param in enumerate(self.parameters):
            grad = param.grad.copy()

            if self.weight_decay != 0:
                grad += self.weight_decay * param.data

            self._v[i] = self.rho * self._v[i] + (1 - self.rho) * (grad ** 2)
            param.data -= self.lr * grad / (np.sqrt(self._v[i]) + self.eps)


class AdaGrad(Optimizer):
    """
    AdaGrad optimizer (Adaptive Gradient).

    Accumulates squared gradients to adapt the learning rate per parameter.
    Works well for sparse gradients but the learning rate can decay to zero.

    G_t = G_{t-1} + grad^2
    param -= lr * grad / (sqrt(G_t) + eps)
    """

    def __init__(self, parameters, lr=0.01, eps=1e-8,
                 weight_decay=0.0, grad_clip=None):
        super().__init__(parameters, lr, grad_clip)
        self.eps = eps
        self.weight_decay = weight_decay
        self._G = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        self._clip_gradients()
        self.t += 1

        for i, param in enumerate(self.parameters):
            grad = param.grad.copy()

            if self.weight_decay != 0:
                grad += self.weight_decay * param.data

            self._G[i] += grad ** 2
            param.data -= self.lr * grad / (np.sqrt(self._G[i]) + self.eps)


# ------------------------------------------------------------------ #
#  Learning Rate Schedulers
# ------------------------------------------------------------------ #

class LRScheduler:
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.step_count = 0

    def step(self):
        self.step_count += 1
        self.optimizer.lr = self.get_lr()

    def get_lr(self):
        raise NotImplementedError


class StepDecay(LRScheduler):
    """
    Multiply learning rate by gamma every step_size epochs.

    lr = base_lr * gamma^(epoch // step_size)
    """

    def __init__(self, optimizer, step_size=30, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self):
        return self.base_lr * (self.gamma ** (self.step_count // self.step_size))


class CosineAnnealing(LRScheduler):
    """
    Cosine annealing learning rate schedule.

    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * t / T_max))

    Smoothly decreases the learning rate following a cosine curve.
    """

    def __init__(self, optimizer, T_max=100, min_lr=0.0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.min_lr = min_lr

    def get_lr(self):
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1 + np.cos(np.pi * self.step_count / self.T_max)
        )


# Registry
OPTIMIZERS = {
    "sgd": SGD,
    "adam": Adam,
    "rmsprop": RMSProp,
    "adagrad": AdaGrad,
}

SCHEDULERS = {
    "step_decay": StepDecay,
    "cosine_annealing": CosineAnnealing,
}
