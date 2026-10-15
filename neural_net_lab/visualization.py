"""
Visualization utilities for neural network training.

Provides publication-quality plots for:
- Decision boundaries
- Loss curves
- Accuracy curves
- Weight distribution histograms
- Gradient flow visualization
- Network architecture diagrams
- Optimizer comparison
- Learning rate schedules
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch


# Custom color palettes
COLORS_CLASSES = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
                  "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"]
CMAP_DECISION = ListedColormap(["#FFE5E5", "#E5FFF5", "#E5F0FF", "#F5FFE5", "#FFF5E5"])
CMAP_BOUNDARY = plt.cm.RdYlBu


def set_style():
    """Apply a clean, modern plotting style."""
    plt.rcParams.update({
        "figure.facecolor": "#FAFAFA",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "text.color": "#333333",
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": "#CCCCCC",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 100,
    })


def plot_decision_boundary(network, X, y, title="Decision Boundary",
                           resolution=200, ax=None, show=False):
    """
    Plot the decision boundary learned by the network.

    Creates a mesh grid over the input space and colors each point
    according to the network's prediction, overlaid with the actual data.

    Args:
        network: Trained Network instance
        X: Input data (N, 2)
        y: Labels (N,)
        title: Plot title
        resolution: Grid resolution
        ax: Matplotlib axes (creates new figure if None)
        show: Whether to call plt.show()
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Define mesh bounds with padding
    margin = 0.5
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Get predictions
    predictions = network.predict(grid)
    if predictions.ndim == 1 or predictions.shape[1] == 1:
        Z = (predictions.ravel() > 0.5).astype(int)
    else:
        Z = np.argmax(predictions, axis=1)
    Z = Z.reshape(xx.shape)

    # Also get raw confidence for contour shading
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        confidence = np.max(predictions.reshape(-1, predictions.shape[1]), axis=1)
    else:
        confidence = np.abs(predictions.ravel() - 0.5) * 2
    confidence = confidence.reshape(xx.shape)

    # Plot decision regions
    n_classes = len(np.unique(y))
    cmap = ListedColormap(COLORS_CLASSES[:n_classes])
    ax.contourf(xx, yy, Z, alpha=0.25, levels=np.arange(-0.5, n_classes + 0.5, 1), cmap=cmap)
    ax.contour(xx, yy, Z, colors="gray", linewidths=0.5, alpha=0.5)

    # Plot data points
    for c in range(n_classes):
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1], c=COLORS_CLASSES[c],
                   edgecolors="white", linewidth=0.5, s=30, label=f"Class {c}",
                   zorder=5)

    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_loss_curves(history, title="Training Progress", ax=None, show=False):
    """
    Plot training and validation loss curves.

    Args:
        history: TrainingHistory instance
        title: Plot title
        ax: Matplotlib axes
        show: Whether to call plt.show()
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    epochs = range(1, len(history.train_loss) + 1)

    ax.plot(epochs, history.train_loss, color="#FF6B6B", linewidth=2, label="Train Loss")
    if history.val_loss:
        ax.plot(epochs, history.val_loss, color="#4ECDC4", linewidth=2, linestyle="--", label="Val Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(1, len(epochs))

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_accuracy_curves(history, title="Accuracy", ax=None, show=False):
    """Plot training and validation accuracy curves."""
    set_style()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    epochs = range(1, len(history.train_acc) + 1)

    ax.plot(epochs, history.train_acc, color="#45B7D1", linewidth=2, label="Train Accuracy")
    if history.val_acc:
        ax.plot(epochs, history.val_acc, color="#96CEB4", linewidth=2, linestyle="--", label="Val Accuracy")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, len(epochs))

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_weight_distributions(network, title="Weight Distributions", show=False):
    """
    Plot histograms of weight values for each layer.

    Useful for diagnosing initialization problems and vanishing/exploding gradients.
    """
    set_style()
    weights = network.get_layer_weights()
    n_layers = len(weights)

    if n_layers == 0:
        return

    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

    for i, (w, b) in enumerate(weights):
        ax = axes[i]
        ax.hist(w.ravel(), bins=50, alpha=0.7, color=colors[i % len(colors)],
                edgecolor="white", linewidth=0.3, density=True)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        mean_val = np.mean(w)
        std_val = np.std(w)
        ax.set_title(f"Layer {i + 1}\n$\\mu$={mean_val:.3f}, $\\sigma$={std_val:.3f}",
                      fontsize=10, fontweight="bold")
        ax.set_xlabel("Weight Value")
        if i == 0:
            ax.set_ylabel("Density")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_gradient_flow(network, title="Gradient Flow", ax=None, show=False):
    """
    Visualize the magnitude of gradients at each layer.

    A bar chart showing average gradient magnitude per layer.
    Healthy training should show roughly consistent gradient magnitudes;
    vanishing gradients appear as bars decreasing toward the input.
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    gradients = network.get_layer_gradients()
    if not gradients:
        return ax

    avg_grads = []
    max_grads = []
    layer_names = []

    for i, (wg, bg) in enumerate(gradients):
        avg_grads.append(np.mean(np.abs(wg)))
        max_grads.append(np.max(np.abs(wg)))
        layer_names.append(f"Layer {i + 1}")

    x = np.arange(len(avg_grads))
    width = 0.35

    bars1 = ax.bar(x - width / 2, max_grads, width, label="Max |grad|",
                   color="#FF6B6B", alpha=0.8, edgecolor="white")
    bars2 = ax.bar(x + width / 2, avg_grads, width, label="Mean |grad|",
                   color="#4ECDC4", alpha=0.8, edgecolor="white")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Gradient Magnitude")
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names)
    ax.legend(fontsize=10)
    ax.set_yscale("log")

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_network_architecture(layer_sizes, title="Network Architecture", ax=None, show=False):
    """
    Draw a diagram of the network architecture.

    Visualizes neurons as circles and connections as lines, with layer sizes
    indicated. Larger layers are capped visually to keep the diagram clean.

    Args:
        layer_sizes: List of integers [input_size, hidden1, hidden2, ..., output_size]
        title: Plot title
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    n_layers = len(layer_sizes)
    max_neurons_draw = 8  # Cap visual neurons per layer

    # Colors for each layer
    layer_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD",
                    "#F7DC6F", "#85C1E9", "#BB8FCE", "#98D8C8"]

    # Positions
    layer_x = np.linspace(0.1, 0.9, n_layers)

    for i, (size, x_pos) in enumerate(zip(layer_sizes, layer_x)):
        n_draw = min(size, max_neurons_draw)
        show_dots = size > max_neurons_draw

        if show_dots:
            n_show = max_neurons_draw - 1  # leave room for "..."
            y_positions = np.linspace(0.15, 0.85, n_show)
        else:
            y_positions = np.linspace(0.15, 0.85, n_draw)

        color = layer_colors[i % len(layer_colors)]

        # Draw connections to next layer
        if i < n_layers - 1:
            next_size = min(layer_sizes[i + 1], max_neurons_draw)
            has_dots_next = layer_sizes[i + 1] > max_neurons_draw
            if has_dots_next:
                next_n = max_neurons_draw - 1
            else:
                next_n = next_size
            next_y = np.linspace(0.15, 0.85, next_n)
            next_x = layer_x[i + 1]

            for y1 in y_positions:
                for y2 in next_y:
                    ax.plot([x_pos, next_x], [y1, y2], color="#CCCCCC",
                            linewidth=0.3, alpha=0.5, zorder=1)

        # Draw neurons
        for y_pos in y_positions:
            circle = plt.Circle((x_pos, y_pos), 0.015, color=color,
                                ec="white", linewidth=1.5, zorder=3)
            ax.add_patch(circle)

        if show_dots:
            ax.text(x_pos, 0.5, "...", ha="center", va="center",
                    fontsize=14, fontweight="bold", color="#666666", zorder=4)

        # Layer label
        label_type = "Input" if i == 0 else ("Output" if i == n_layers - 1 else f"Hidden {i}")
        ax.text(x_pos, 0.02, f"{label_type}\n({size})", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#333333")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontweight="bold", fontsize=14, pad=10)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_optimizer_comparison(results, dataset_name="", show=False):
    """
    Compare training curves of different optimizers side by side.

    Args:
        results: Dict mapping optimizer name to TrainingHistory
        dataset_name: Name of the dataset for the title
        show: Whether to show immediately
    """
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"SGD": "#FF6B6B", "Adam": "#4ECDC4", "RMSProp": "#45B7D1", "AdaGrad": "#96CEB4"}

    for name, history in results.items():
        color = colors.get(name, "#888888")
        epochs = range(1, len(history.train_loss) + 1)

        axes[0].plot(epochs, history.train_loss, label=name, linewidth=2, color=color)
        if history.train_acc:
            axes[1].plot(epochs, history.train_acc, label=name, linewidth=2, color=color)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss Comparison{' - ' + dataset_name if dataset_name else ''}", fontweight="bold")
    axes[0].legend(fontsize=10)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Accuracy Comparison{' - ' + dataset_name if dataset_name else ''}", fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_training_dashboard(network, history, X, y, dataset_name=""):
    """
    Create a comprehensive training dashboard with multiple panels.

    Combines decision boundary, loss curve, accuracy curve, and gradient flow
    into a single figure.

    Args:
        network: Trained Network instance
        history: TrainingHistory from training
        X: Input data (N, 2) -- only for 2D datasets
        y: Labels
        dataset_name: Name for display

    Returns:
        matplotlib Figure
    """
    set_style()

    is_2d = X.shape[1] == 2

    if is_2d:
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

        # Decision boundary (large panel)
        ax_db = fig.add_subplot(gs[0, :2])
        plot_decision_boundary(network, X, y,
                               title=f"Decision Boundary - {dataset_name}", ax=ax_db)

        # Network architecture
        layer_sizes = []
        from .layers import Dense
        for layer in network.layers:
            if isinstance(layer, Dense):
                if not layer_sizes:
                    layer_sizes.append(layer.in_features)
                layer_sizes.append(layer.out_features)
        if layer_sizes:
            ax_arch = fig.add_subplot(gs[0, 2])
            plot_network_architecture(layer_sizes, title="Architecture", ax=ax_arch)

        # Loss curve
        ax_loss = fig.add_subplot(gs[1, 0])
        plot_loss_curves(history, title="Loss", ax=ax_loss)

        # Accuracy curve
        ax_acc = fig.add_subplot(gs[1, 1])
        plot_accuracy_curves(history, title="Accuracy", ax=ax_acc)

        # Gradient flow
        ax_grad = fig.add_subplot(gs[1, 2])
        plot_gradient_flow(network, title="Gradient Flow", ax=ax_grad)

    else:
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

        # Loss curve
        ax_loss = fig.add_subplot(gs[0, 0])
        plot_loss_curves(history, title="Loss", ax=ax_loss)

        # Accuracy curve
        ax_acc = fig.add_subplot(gs[0, 1])
        plot_accuracy_curves(history, title="Accuracy", ax=ax_acc)

        # Gradient flow
        ax_grad = fig.add_subplot(gs[1, 0])
        plot_gradient_flow(network, title="Gradient Flow", ax=ax_grad)

        # Network architecture
        layer_sizes = []
        from .layers import Dense
        for layer in network.layers:
            if isinstance(layer, Dense):
                if not layer_sizes:
                    layer_sizes.append(layer.in_features)
                layer_sizes.append(layer.out_features)
        if layer_sizes:
            ax_arch = fig.add_subplot(gs[1, 1])
            plot_network_architecture(layer_sizes, title="Architecture", ax=ax_arch)

    fig.suptitle(f"Neural Network Training Dashboard - {dataset_name}",
                 fontsize=16, fontweight="bold", y=1.01)

    return fig


def plot_decision_boundary_evolution(network_snapshots, X, y, dataset_name="",
                                     show=False):
    """
    Show how the decision boundary evolves during training.

    Args:
        network_snapshots: List of (epoch, network) tuples
        X: Input data (N, 2)
        y: Labels

    Returns:
        matplotlib Figure
    """
    set_style()
    n = len(network_snapshots)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (epoch, net) in enumerate(network_snapshots):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        plot_decision_boundary(net, X, y,
                               title=f"Epoch {epoch}", ax=ax)

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    fig.suptitle(f"Decision Boundary Evolution - {dataset_name}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_learning_rate_schedule(scheduler_cls, optimizer_lr, total_epochs,
                                scheduler_kwargs=None, title=None, ax=None, show=False):
    """
    Visualize a learning rate schedule over epochs.

    Args:
        scheduler_cls: Scheduler class
        optimizer_lr: Base learning rate
        total_epochs: Number of epochs to simulate
        scheduler_kwargs: kwargs for the scheduler
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Create a dummy optimizer-like object
    class DummyOpt:
        def __init__(self, lr):
            self.lr = lr

    opt = DummyOpt(optimizer_lr)
    kwargs = scheduler_kwargs or {}
    scheduler = scheduler_cls(opt, **kwargs)

    lrs = []
    for epoch in range(total_epochs):
        lrs.append(opt.lr)
        scheduler.step()

    ax.plot(range(1, total_epochs + 1), lrs, color="#FF6B6B", linewidth=2)
    ax.fill_between(range(1, total_epochs + 1), lrs, alpha=0.15, color="#FF6B6B")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title(title or f"{scheduler_cls.__name__} Schedule", fontweight="bold")

    if show:
        plt.tight_layout()
        plt.show()

    return ax
