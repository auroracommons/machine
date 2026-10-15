"""
Gradient checks for the core layers (Dense, BatchNorm, Dropout).

Verifies analytical gradients derived in neural_net_lab/layers.py
against central finite differences:

    numeric_grad = (f(x + eps) - f(x - eps)) / (2 * eps)

and asserts the relative error

    |analytic - numeric| / max(1e-8, |analytic| + |numeric|)

stays below 1e-5 (or absolute floor 1e-7) for all input and parameter
elements on small test configurations.

Fulfills Good First Issue #1 from CONTRIBUTING.md.

Run: python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_net_lab.layers import Dense, BatchNorm, Dropout
from neural_net_lab.network import Network

EPS = 1e-6
TOL = 1e-5
ABS_TOL = 1e-7


def numeric_grad(scalar_fn, arr, eps=EPS):
    grad = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = arr[idx]
        arr[idx] = old + eps
        f_plus = scalar_fn()
        arr[idx] = old - eps
        f_minus = scalar_fn()
        arr[idx] = old
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def assert_grad_close(test_case, analytic, numeric, label):
    denom = np.maximum(1e-8, np.abs(analytic) + np.abs(numeric))
    rel = np.abs(analytic - numeric) / denom
    abs_err = np.abs(analytic - numeric).max()
    test_case.assertTrue(
        rel.max() < TOL or abs_err < ABS_TOL,
        f"{label}: max relative error {rel.max():.3e}, "
        f"max absolute error {abs_err:.3e} (analytic vs numeric mismatch)",
    )


def check_layer(test_case, layer, x, seed=0):
    """Full gradient check of one Layer: input grad + all parameter grads."""
    rng = np.random.default_rng(seed)
    out = layer.forward(x)
    G = rng.standard_normal(out.shape)

    def scalar_fn():
        return float(np.sum(layer.forward(x) * G))

    for p in layer.parameters():
        p.zero_grad()
    analytic_x = layer.backward(G)
    analytic_params = [p.grad.copy() for p in layer.parameters()]

    numeric_x = numeric_grad(scalar_fn, x)
    assert_grad_close(test_case, analytic_x, numeric_x, "input")
    layer.forward(x)

    for i, p in enumerate(layer.parameters()):
        analytic = analytic_params[i]
        numeric = numeric_grad(scalar_fn, p.data)
        assert_grad_close(
            test_case, analytic, numeric, f"param[{i}] shape {p.data.shape}"
        )
        layer.forward(x)


class TestDenseGradients(unittest.TestCase):
    def test_linear_with_bias(self):
        rng = np.random.default_rng(101)
        layer = Dense(5, 3, activation=None, use_bias=True)
        x = rng.standard_normal((4, 5))
        check_layer(self, layer, x, seed=101)

    def test_linear_without_bias(self):
        rng = np.random.default_rng(102)
        layer = Dense(4, 3, activation=None, use_bias=False)
        x = rng.standard_normal((3, 4))
        check_layer(self, layer, x, seed=102)

    def test_with_relu_activation(self):
        rng = np.random.default_rng(103)
        layer = Dense(4, 3, activation="relu", use_bias=True)
        # Offset inputs slightly away from zero to avoid non-differentiable cusp
        x = rng.standard_normal((4, 4)) + 0.5
        check_layer(self, layer, x, seed=103)

    def test_with_tanh_activation(self):
        rng = np.random.default_rng(104)
        layer = Dense(3, 3, activation="tanh", use_bias=True)
        x = rng.standard_normal((3, 3))
        check_layer(self, layer, x, seed=104)

    def test_with_sigmoid_activation(self):
        rng = np.random.default_rng(105)
        layer = Dense(3, 4, activation="sigmoid", use_bias=True)
        x = rng.standard_normal((3, 3))
        check_layer(self, layer, x, seed=105)


class TestBatchNormGradients(unittest.TestCase):
    def test_training_gradients(self):
        rng = np.random.default_rng(201)
        layer = BatchNorm(4)
        layer.gamma.data = rng.standard_normal(4)
        layer.beta.data = rng.standard_normal(4)
        x = rng.standard_normal((6, 4))
        check_layer(self, layer, x, seed=201)


class TestDropoutGradients(unittest.TestCase):
    def test_training_mode(self):
        rng = np.random.default_rng(301)
        layer = Dropout(p=0.4)
        x = rng.standard_normal((4, 5))
        # Forward pass generates and caches self._mask
        out = layer.forward(x)
        G = rng.standard_normal(out.shape)
        analytic_x = layer.backward(G)

        # Numerical gradient holding the cached dropout mask fixed
        mask = layer._mask

        def scalar_fn():
            return float(np.sum((x * mask) * G))

        numeric_x = numeric_grad(scalar_fn, x)
        assert_grad_close(self, analytic_x, numeric_x, "dropout_training")

    def test_eval_mode(self):
        rng = np.random.default_rng(302)
        layer = Dropout(p=0.5)
        layer.eval()
        x = rng.standard_normal((4, 5))
        out = layer.forward(x)
        G = rng.standard_normal(out.shape)
        analytic_x = layer.backward(G)

        def scalar_fn():
            return float(np.sum(layer.forward(x) * G))

        numeric_x = numeric_grad(scalar_fn, x)
        assert_grad_close(self, analytic_x, numeric_x, "dropout_eval")


class TestNetworkTrainingSmoke(unittest.TestCase):
    def test_loss_decreases_on_toy_dataset(self):
        rng = np.random.default_rng(401)
        X = rng.standard_normal((32, 4))
        y = (np.sum(X, axis=1) > 0).astype(int)

        net = Network()
        net.add(Dense(4, 8, activation="relu"))
        net.add(BatchNorm(8))
        net.add(Dense(8, 2))
        net.compile(optimizer="adam", loss="cross_entropy", lr=0.05)
        history = net.fit(X, y, epochs=25, batch_size=32, verbose=False)

        self.assertLess(
            history.train_loss[-1],
            history.train_loss[0],
            f"Training loss did not decrease: {history.train_loss[0]:.4f} -> {history.train_loss[-1]:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
