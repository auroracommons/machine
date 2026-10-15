"""
Loss functions for neural network training.

Each loss computes both the scalar loss value and the gradient of the loss
with respect to the predictions, which seeds the backward pass.
"""

import numpy as np


class Loss:
    """Base class for loss functions."""

    def forward(self, predictions, targets):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)


class MSELoss(Loss):
    """
    Mean Squared Error loss.

    L = (1/N) * sum((predictions - targets)^2)

    Commonly used for regression tasks.
    """

    def forward(self, predictions, targets):
        self._predictions = predictions
        self._targets = targets
        self._diff = predictions - targets
        return np.mean(self._diff ** 2)

    def backward(self):
        N = self._predictions.shape[0]
        return 2.0 * self._diff / N


class CrossEntropyLoss(Loss):
    """
    Cross-Entropy loss for multi-class classification.

    Combines softmax and negative log-likelihood:
    L = -(1/N) * sum(targets * log(softmax(logits)))

    Expects:
      - predictions: raw logits of shape (N, C) where C is number of classes
      - targets: one-hot encoded targets of shape (N, C) OR class indices of shape (N,)
    """

    def forward(self, predictions, targets):
        self._logits = predictions
        N = predictions.shape[0]

        # Convert integer labels to one-hot if needed
        if targets.ndim == 1:
            self._targets_onehot = np.zeros_like(predictions)
            self._targets_onehot[np.arange(N), targets.astype(int)] = 1.0
        else:
            self._targets_onehot = targets

        # Numerically stable softmax
        shifted = predictions - predictions.max(axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        self._probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        # Negative log-likelihood
        log_probs = np.log(np.clip(self._probs, 1e-12, 1.0))
        loss = -np.sum(self._targets_onehot * log_probs) / N
        return loss

    def backward(self):
        N = self._logits.shape[0]
        # The gradient of cross-entropy w.r.t. logits is simply: (softmax - targets) / N
        return (self._probs - self._targets_onehot) / N


class BinaryCrossEntropyLoss(Loss):
    """
    Binary Cross-Entropy loss.

    L = -(1/N) * sum(t * log(p) + (1-t) * log(1-p))

    Expects:
      - predictions: probabilities in [0, 1] (after sigmoid), shape (N, 1) or (N,)
      - targets: binary labels 0 or 1, same shape as predictions
    """

    def forward(self, predictions, targets):
        self._predictions = np.clip(predictions, 1e-12, 1.0 - 1e-12)
        self._targets = targets
        N = predictions.shape[0]

        loss = -np.mean(
            targets * np.log(self._predictions) +
            (1.0 - targets) * np.log(1.0 - self._predictions)
        )
        return loss

    def backward(self):
        N = self._predictions.shape[0]
        grad = (-(self._targets / self._predictions) +
                (1.0 - self._targets) / (1.0 - self._predictions))
        return grad / N


# Registry
LOSSES = {
    "mse": MSELoss,
    "cross_entropy": CrossEntropyLoss,
    "binary_cross_entropy": BinaryCrossEntropyLoss,
}


def get_loss(name):
    """Get a loss function by name."""
    if isinstance(name, Loss):
        return name
    name = name.lower()
    if name not in LOSSES:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSSES.keys())}")
    return LOSSES[name]()
