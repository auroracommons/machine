"""
Neural Network Laboratory - Demo Script

Trains neural networks on several toy datasets and produces visualizations.
Run with: python -m neural_net_lab.main  (from the project root)
Or:       python main.py                  (from the neural-net-lab directory)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import os
import time

# Allow running as both `python main.py` and `python -m neural_net_lab.main`
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_net_lab.layers import Dense, Dropout, BatchNorm
from neural_net_lab.network import Network, PrintProgress
from neural_net_lab.optimizers import StepDecay, CosineAnnealing
from neural_net_lab.datasets import (
    make_spirals, make_moons, make_circles, make_xor,
    make_gaussian_blobs, make_mnist_subset,
)
from neural_net_lab.visualization import (
    plot_decision_boundary, plot_loss_curves, plot_accuracy_curves,
    plot_weight_distributions, plot_gradient_flow,
    plot_network_architecture, plot_optimizer_comparison,
    plot_training_dashboard, plot_learning_rate_schedule,
    set_style,
)


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Saved: {path}")


# ------------------------------------------------------------------ #
#  Demo 1: Spirals - The classic non-linear challenge
# ------------------------------------------------------------------ #

def demo_spirals():
    print("\n" + "=" * 60)
    print("DEMO 1: Spiral Dataset")
    print("=" * 60)

    np.random.seed(42)
    X, y = make_spirals(n_samples=400, n_classes=3, noise=0.15, random_state=42)

    net = Network()
    net.add(Dense(2, 64, activation="relu", weight_init="he"))
    net.add(Dense(64, 64, activation="relu", weight_init="he"))
    net.add(Dense(64, 3, weight_init="xavier"))
    net.compile(optimizer="adam", loss="cross_entropy", lr=0.01, grad_clip=5.0)

    print("  Network architecture:")
    net.summary()

    history = net.fit(X, y, epochs=200, batch_size=32, verbose=True)

    final_preds = net.predict(X)
    final_acc = np.mean(np.argmax(final_preds, axis=1) == y)
    print(f"  Final accuracy: {final_acc:.4f}")

    # Dashboard
    fig = plot_training_dashboard(net, history, X, y, dataset_name="Spirals (3 classes)")
    save_fig(fig, "01_spirals_dashboard.png")

    # Weight distributions
    fig = plot_weight_distributions(net, title="Spirals - Weight Distributions")
    save_fig(fig, "02_spirals_weights.png")


# ------------------------------------------------------------------ #
#  Demo 2: Moons - Binary classification
# ------------------------------------------------------------------ #

def demo_moons():
    print("\n" + "=" * 60)
    print("DEMO 2: Moons Dataset")
    print("=" * 60)

    np.random.seed(123)
    X, y = make_moons(n_samples=400, noise=0.15, random_state=123)

    net = Network()
    net.add(Dense(2, 32, activation="relu"))
    net.add(Dense(32, 16, activation="relu"))
    net.add(Dense(16, 2))
    net.compile(optimizer="adam", loss="cross_entropy", lr=0.01)

    history = net.fit(X, y, epochs=150, batch_size=32, verbose=True)

    final_preds = net.predict(X)
    final_acc = np.mean(np.argmax(final_preds, axis=1) == y)
    print(f"  Final accuracy: {final_acc:.4f}")

    fig = plot_training_dashboard(net, history, X, y, dataset_name="Moons")
    save_fig(fig, "03_moons_dashboard.png")


# ------------------------------------------------------------------ #
#  Demo 3: XOR - The problem that needs hidden layers
# ------------------------------------------------------------------ #

def demo_xor():
    print("\n" + "=" * 60)
    print("DEMO 3: XOR Dataset")
    print("=" * 60)

    np.random.seed(7)
    X, y = make_xor(n_samples=400, noise=0.12, random_state=7)

    net = Network()
    net.add(Dense(2, 32, activation="relu"))
    net.add(Dense(32, 16, activation="relu"))
    net.add(Dense(16, 2))
    net.compile(optimizer="adam", loss="cross_entropy", lr=0.005)

    history = net.fit(X, y, epochs=200, batch_size=32, verbose=True)

    fig = plot_training_dashboard(net, history, X, y, dataset_name="XOR")
    save_fig(fig, "04_xor_dashboard.png")


# ------------------------------------------------------------------ #
#  Demo 4: Circles with BatchNorm and Dropout
# ------------------------------------------------------------------ #

def demo_circles_regularized():
    print("\n" + "=" * 60)
    print("DEMO 4: Circles with BatchNorm & Dropout")
    print("=" * 60)

    np.random.seed(99)
    X, y = make_circles(n_samples=400, noise=0.08, factor=0.4, random_state=99)

    net = Network()
    net.add(Dense(2, 32, activation="relu"))
    net.add(BatchNorm(32))
    net.add(Dropout(0.2))
    net.add(Dense(32, 16, activation="relu"))
    net.add(BatchNorm(16))
    net.add(Dense(16, 2))
    net.compile(optimizer="adam", loss="cross_entropy", lr=0.01)

    print("  Network with BatchNorm + Dropout:")
    net.summary()

    history = net.fit(X, y, epochs=150, batch_size=32, verbose=True)

    fig = plot_training_dashboard(net, history, X, y,
                                   dataset_name="Circles (BatchNorm + Dropout)")
    save_fig(fig, "05_circles_regularized_dashboard.png")


# ------------------------------------------------------------------ #
#  Demo 5: Optimizer Comparison on Spirals
# ------------------------------------------------------------------ #

def demo_optimizer_comparison():
    print("\n" + "=" * 60)
    print("DEMO 5: Optimizer Comparison")
    print("=" * 60)

    np.random.seed(42)
    X, y = make_spirals(n_samples=400, n_classes=2, noise=0.15, random_state=42)

    optimizers_config = {
        "SGD": {"optimizer": "sgd", "lr": 0.1, "momentum": 0.9},
        "Adam": {"optimizer": "adam", "lr": 0.01},
        "RMSProp": {"optimizer": "rmsprop", "lr": 0.005},
        "AdaGrad": {"optimizer": "adagrad", "lr": 0.05},
    }

    results = {}
    epochs = 150

    for name, config in optimizers_config.items():
        print(f"\n  Training with {name}...")
        np.random.seed(42)  # Same initialization for fair comparison

        net = Network()
        net.add(Dense(2, 48, activation="relu", weight_init="he"))
        net.add(Dense(48, 24, activation="relu", weight_init="he"))
        net.add(Dense(24, 2, weight_init="xavier"))

        cfg = dict(config)  # copy to avoid mutating the original
        opt_name = cfg.pop("optimizer")
        lr = cfg.pop("lr")
        net.compile(optimizer=opt_name, loss="cross_entropy", lr=lr, **cfg)

        history = net.fit(X, y, epochs=epochs, batch_size=32, verbose=True)
        results[name] = history

        final_preds = net.predict(X)
        acc = np.mean(np.argmax(final_preds, axis=1) == y)
        print(f"    {name} final accuracy: {acc:.4f}")

    fig = plot_optimizer_comparison(results, dataset_name="Spirals (2-class)")
    save_fig(fig, "06_optimizer_comparison.png")


# ------------------------------------------------------------------ #
#  Demo 6: Learning Rate Schedules
# ------------------------------------------------------------------ #

def demo_lr_schedules():
    print("\n" + "=" * 60)
    print("DEMO 6: Learning Rate Schedules")
    print("=" * 60)

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_learning_rate_schedule(
        StepDecay, optimizer_lr=0.1, total_epochs=200,
        scheduler_kwargs={"step_size": 50, "gamma": 0.5},
        title="Step Decay (step=50, gamma=0.5)", ax=axes[0]
    )
    plot_learning_rate_schedule(
        CosineAnnealing, optimizer_lr=0.1, total_epochs=200,
        scheduler_kwargs={"T_max": 200, "min_lr": 0.001},
        title="Cosine Annealing (T_max=200)", ax=axes[1]
    )

    fig.suptitle("Learning Rate Schedules", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "07_lr_schedules.png")


# ------------------------------------------------------------------ #
#  Demo 7: MNIST-like digit classification
# ------------------------------------------------------------------ #

def demo_mnist():
    print("\n" + "=" * 60)
    print("DEMO 7: Procedural MNIST-like Digits")
    print("=" * 60)

    np.random.seed(42)
    X, y = make_mnist_subset(n_samples=600, digits=[0, 1, 2, 3, 4],
                              image_size=8, random_state=42)

    # Normalize
    X = X / (X.max() + 1e-8)

    n_features = X.shape[1]  # 64 for 8x8
    n_classes = 5

    net = Network()
    net.add(Dense(n_features, 64, activation="relu"))
    net.add(Dense(64, 32, activation="relu"))
    net.add(Dense(32, n_classes))
    net.compile(optimizer="adam", loss="cross_entropy", lr=0.005)

    print(f"  Input features: {n_features}, Classes: {n_classes}")
    net.summary()

    history = net.fit(X, y, epochs=100, batch_size=32, verbose=True)

    final_preds = net.predict(X)
    acc = np.mean(np.argmax(final_preds, axis=1) == y)
    print(f"  Final accuracy: {acc:.4f}")

    # Training curves figure (no decision boundary for high-dim data)
    fig = plot_training_dashboard(net, history, X, y, dataset_name="Procedural Digits")
    save_fig(fig, "08_mnist_dashboard.png")

    # Show some sample digits
    set_style()
    fig_samples, axes_s = plt.subplots(2, 10, figsize=(14, 3.5))
    fig_samples.suptitle("Sample Generated Digits (Procedural MNIST-like)",
                          fontweight="bold", fontsize=13)
    for row in range(2):
        for col in range(10):
            idx = row * 10 + col
            ax = axes_s[row, col]
            img = X[idx].reshape(8, 8)
            ax.imshow(img, cmap="gray_r", interpolation="nearest")
            pred = np.argmax(net.predict(X[idx:idx + 1]))
            true = y[idx]
            color = "#2ECC71" if pred == true else "#E74C3C"
            ax.set_title(f"{pred}", fontsize=9, color=color, fontweight="bold")
            ax.axis("off")
    plt.tight_layout()
    save_fig(fig_samples, "09_digit_samples.png")


# ------------------------------------------------------------------ #
#  Demo 8: Activation function comparison
# ------------------------------------------------------------------ #

def demo_activation_comparison():
    print("\n" + "=" * 60)
    print("DEMO 8: Activation Function Comparison")
    print("=" * 60)

    np.random.seed(42)
    X, y = make_spirals(n_samples=400, n_classes=2, noise=0.15, random_state=42)

    activations_to_test = ["relu", "tanh", "sigmoid", "gelu"]
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes_flat = axes.ravel()

    for idx, act_name in enumerate(activations_to_test):
        print(f"\n  Training with {act_name}...")
        np.random.seed(42)

        net = Network()
        net.add(Dense(2, 48, activation=act_name, weight_init="he"))
        net.add(Dense(48, 24, activation=act_name, weight_init="he"))
        net.add(Dense(24, 2, weight_init="xavier"))
        net.compile(optimizer="adam", loss="cross_entropy", lr=0.01)

        history = net.fit(X, y, epochs=150, batch_size=32, verbose=True)

        final_preds = net.predict(X)
        acc = np.mean(np.argmax(final_preds, axis=1) == y)

        plot_decision_boundary(net, X, y,
                               title=f"{act_name.upper()} (acc={acc:.3f})",
                               ax=axes_flat[idx])

    fig.suptitle("Activation Function Comparison - Spirals",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, "10_activation_comparison.png")


# ------------------------------------------------------------------ #
#  Demo 9: Network architecture diagram
# ------------------------------------------------------------------ #

def demo_architecture_diagram():
    print("\n" + "=" * 60)
    print("DEMO 9: Architecture Diagram")
    print("=" * 60)

    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_network_architecture([2, 16, 8, 2], title="Small: 2-16-8-2", ax=axes[0])
    plot_network_architecture([2, 64, 64, 32, 3], title="Medium: 2-64-64-32-3", ax=axes[1])
    plot_network_architecture([64, 128, 64, 32, 5], title="Deep: 64-128-64-32-5", ax=axes[2])

    fig.suptitle("Network Architecture Diagrams", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, "11_architecture_diagrams.png")


# ------------------------------------------------------------------ #
#  Demo 10: Weight initialization comparison
# ------------------------------------------------------------------ #

def demo_weight_init():
    print("\n" + "=" * 60)
    print("DEMO 10: Weight Initialization Comparison")
    print("=" * 60)

    np.random.seed(42)
    X, y = make_moons(n_samples=300, noise=0.12, random_state=42)

    inits = ["he", "xavier", "normal"]
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, init_name in enumerate(inits):
        print(f"\n  Training with {init_name} initialization...")
        np.random.seed(42)

        net = Network()
        net.add(Dense(2, 48, activation="relu", weight_init=init_name))
        net.add(Dense(48, 48, activation="relu", weight_init=init_name))
        net.add(Dense(48, 24, activation="relu", weight_init=init_name))
        net.add(Dense(24, 2, weight_init=init_name))
        net.compile(optimizer="adam", loss="cross_entropy", lr=0.01)

        history = net.fit(X, y, epochs=100, batch_size=32, verbose=True)

        final_preds = net.predict(X)
        acc = np.mean(np.argmax(final_preds, axis=1) == y)

        plot_decision_boundary(net, X, y,
                               title=f"{init_name.title()} Init (acc={acc:.3f})",
                               ax=axes[idx])

    fig.suptitle("Weight Initialization Comparison - Moons",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, "12_weight_init_comparison.png")


# ================================================================== #
#  Main entry point
# ================================================================== #

def main():
    print()
    print("*" * 60)
    print("   NEURAL NETWORK LABORATORY")
    print("   From-Scratch Implementation with Visualizer")
    print("*" * 60)

    ensure_output_dir()
    t0 = time.time()

    demo_spirals()
    demo_moons()
    demo_xor()
    demo_circles_regularized()
    demo_optimizer_comparison()
    demo_lr_schedules()
    demo_mnist()
    demo_activation_comparison()
    demo_architecture_diagram()
    demo_weight_init()

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  All demos complete! Total time: {elapsed:.1f}s")
    print(f"  Figures saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
