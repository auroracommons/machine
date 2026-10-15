"""
Built-in toy datasets for testing and visualization.

All datasets are generated procedurally with numpy -- no file downloads required.
Each function returns (X, y) where X is shape (N, features) and y contains labels.
"""

import numpy as np


def make_spirals(n_samples=300, n_classes=2, noise=0.2, random_state=None):
    """
    Generate a spiral dataset.

    Creates interleaving spirals -- a classic hard problem for linear classifiers
    that requires non-linear decision boundaries.

    Args:
        n_samples: Total number of samples (split evenly across classes)
        n_classes: Number of spiral arms (classes)
        noise: Standard deviation of Gaussian noise
        random_state: Random seed for reproducibility

    Returns:
        X: (n_samples, 2) array of 2D coordinates
        y: (n_samples,) array of class labels (integers)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_per_class = n_samples // n_classes
    X = np.zeros((n_samples, 2))
    y = np.zeros(n_samples, dtype=int)

    for j in range(n_classes):
        ix = range(n_per_class * j, n_per_class * (j + 1))
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(j * 4.0, (j + 1) * 4.0, n_per_class) + np.random.randn(n_per_class) * noise
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    return X, y


def make_moons(n_samples=300, noise=0.1, random_state=None):
    """
    Generate two interleaving half-moon shapes.

    A popular 2D binary classification dataset that requires a non-linear
    decision boundary.

    Args:
        n_samples: Total number of samples
        noise: Standard deviation of Gaussian noise
        random_state: Random seed

    Returns:
        X: (n_samples, 2) coordinates
        y: (n_samples,) binary labels
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_half = n_samples // 2

    # Upper moon
    theta1 = np.linspace(0, np.pi, n_half)
    x1 = np.cos(theta1)
    y1 = np.sin(theta1)

    # Lower moon (shifted)
    theta2 = np.linspace(0, np.pi, n_samples - n_half)
    x2 = 1 - np.cos(theta2)
    y2 = 1 - np.sin(theta2) - 0.5

    X = np.vstack([np.c_[x1, y1], np.c_[x2, y2]])
    y = np.hstack([np.zeros(n_half), np.ones(n_samples - n_half)]).astype(int)

    # Add noise
    X += np.random.randn(*X.shape) * noise

    return X, y


def make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=None):
    """
    Generate concentric circles dataset.

    An inner circle surrounded by an outer circle -- linearly inseparable.

    Args:
        n_samples: Total number of samples
        noise: Standard deviation of noise
        factor: Ratio of inner circle radius to outer circle radius
        random_state: Random seed

    Returns:
        X: (n_samples, 2) coordinates
        y: (n_samples,) binary labels (0 = outer, 1 = inner)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    # Outer circle
    theta_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    X_outer = np.c_[np.cos(theta_outer), np.sin(theta_outer)]

    # Inner circle
    theta_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
    X_inner = np.c_[np.cos(theta_inner) * factor, np.sin(theta_inner) * factor]

    X = np.vstack([X_outer, X_inner])
    y = np.hstack([np.zeros(n_outer), np.ones(n_inner)]).astype(int)

    X += np.random.randn(*X.shape) * noise
    return X, y


def make_xor(n_samples=300, noise=0.15, random_state=None):
    """
    Generate an XOR pattern dataset.

    Four clusters arranged in an XOR pattern. Points in opposite corners
    share the same class. The classic example of a problem requiring at least
    one hidden layer.

    Args:
        n_samples: Total number of samples
        noise: Standard deviation of noise around cluster centers
        random_state: Random seed

    Returns:
        X: (n_samples, 2) coordinates
        y: (n_samples,) binary labels
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_per_cluster = n_samples // 4

    centers = np.array([
        [0, 0],
        [1, 1],
        [0, 1],
        [1, 0],
    ], dtype=np.float64)

    labels = [0, 0, 1, 1]  # XOR pattern

    X_list = []
    y_list = []
    for i, (center, label) in enumerate(zip(centers, labels)):
        n = n_per_cluster if i < 3 else n_samples - 3 * n_per_cluster
        X_list.append(center + np.random.randn(n, 2) * noise)
        y_list.append(np.full(n, label))

    X = np.vstack(X_list)
    y = np.hstack(y_list).astype(int)

    return X, y


def make_gaussian_blobs(n_samples=300, n_classes=3, cluster_std=0.5, random_state=None):
    """
    Generate isotropic Gaussian blobs for clustering/classification.

    Args:
        n_samples: Total number of samples
        n_classes: Number of blobs (classes)
        cluster_std: Standard deviation of each blob
        random_state: Random seed

    Returns:
        X: (n_samples, 2) coordinates
        y: (n_samples,) class labels
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_per = n_samples // n_classes
    # Place centers on a circle for nice spacing
    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
    centers = 2.0 * np.c_[np.cos(angles), np.sin(angles)]

    X_list = []
    y_list = []
    for i, center in enumerate(centers):
        n = n_per if i < n_classes - 1 else n_samples - (n_classes - 1) * n_per
        X_list.append(center + np.random.randn(n, 2) * cluster_std)
        y_list.append(np.full(n, i))

    X = np.vstack(X_list)
    y = np.hstack(y_list).astype(int)

    return X, y


def make_mnist_subset(n_samples=500, digits=None, image_size=8, random_state=None):
    """
    Generate a procedural MNIST-like digit dataset.

    Creates simplified digit images procedurally using basic tools for drawing.
    Not real MNIST, but mimics its structure for testing multi-class classifiers.

    Args:
        n_samples: Total number of samples
        digits: List of digits to include (default: [0, 1, 2, 3, 4])
        image_size: Size of each image (image_size x image_size)
        random_state: Random seed

    Returns:
        X: (n_samples, image_size*image_size) flattened images
        y: (n_samples,) digit labels
    """
    if random_state is not None:
        np.random.seed(random_state)

    if digits is None:
        digits = [0, 1, 2, 3, 4]

    n_classes = len(digits)
    n_per = n_samples // n_classes

    def draw_digit(digit, size):
        """Draw a simplified digit on a grid."""
        img = np.zeros((size, size))
        mid = size // 2
        q1 = size // 4
        q3 = 3 * size // 4

        if digit == 0:
            # Circle / oval
            for i in range(size):
                for j in range(size):
                    r = np.sqrt(((i - mid) / (mid - 1)) ** 2 + ((j - mid) / (mid - 1)) ** 2)
                    if 0.5 < r < 1.0:
                        img[i, j] = 1.0
        elif digit == 1:
            # Vertical line
            img[q1:q3 + 1, mid] = 1.0
            img[q1:q3 + 1, mid - 1] = 0.5
        elif digit == 2:
            # Top bar, diagonal, bottom bar
            img[q1, q1:q3] = 1.0
            img[q1 + 1, q1:q3] = 0.5
            for k in range(q3 - q1):
                r = q1 + k
                c = q3 - 1 - int(k * (q3 - q1 - 1) / max(q3 - q1 - 1, 1))
                if 0 <= r < size and 0 <= c < size:
                    img[r, c] = 1.0
            img[q3, q1:q3] = 1.0
            img[q3 - 1, q1:q3] = 0.5
        elif digit == 3:
            # Three horizontal bars connected on right
            img[q1, q1:q3] = 1.0
            img[mid, q1:q3] = 1.0
            img[q3, q1:q3] = 1.0
            img[q1:q3 + 1, q3 - 1] = 1.0
        elif digit == 4:
            # Top-left vertical + horizontal middle + right vertical
            img[q1:mid + 1, q1] = 1.0
            img[mid, q1:q3] = 1.0
            img[q1:q3 + 1, q3 - 1] = 1.0
        elif digit == 5:
            img[q1, q1:q3] = 1.0
            img[q1:mid + 1, q1] = 1.0
            img[mid, q1:q3] = 1.0
            img[mid:q3 + 1, q3 - 1] = 1.0
            img[q3, q1:q3] = 1.0
        elif digit == 6:
            img[q1, q1:q3] = 1.0
            img[q1:q3 + 1, q1] = 1.0
            img[mid, q1:q3] = 1.0
            img[mid:q3 + 1, q3 - 1] = 1.0
            img[q3, q1:q3] = 1.0
        elif digit == 7:
            img[q1, q1:q3] = 1.0
            img[q1:q3 + 1, q3 - 1] = 1.0
        elif digit == 8:
            img[q1, q1:q3] = 1.0
            img[mid, q1:q3] = 1.0
            img[q3, q1:q3] = 1.0
            img[q1:q3 + 1, q1] = 1.0
            img[q1:q3 + 1, q3 - 1] = 1.0
        elif digit == 9:
            img[q1, q1:q3] = 1.0
            img[q1:mid + 1, q1] = 1.0
            img[mid, q1:q3] = 1.0
            img[q1:q3 + 1, q3 - 1] = 1.0

        return img

    X_list = []
    y_list = []

    for idx, digit in enumerate(digits):
        n = n_per if idx < n_classes - 1 else n_samples - (n_classes - 1) * n_per
        base_img = draw_digit(digit, image_size)

        for _ in range(n):
            # Add random variations: noise, slight shifts, intensity variation
            img = base_img.copy()

            # Random shift (0-1 pixel)
            shift_r = np.random.randint(-1, 2)
            shift_c = np.random.randint(-1, 2)
            img = np.roll(np.roll(img, shift_r, axis=0), shift_c, axis=1)

            # Add noise
            img += np.random.randn(image_size, image_size) * 0.15

            # Random intensity
            img *= (0.7 + 0.3 * np.random.rand())

            img = np.clip(img, 0, 1)
            X_list.append(img.ravel())
            y_list.append(idx)  # Use index as label for clean 0-based classes

    X = np.array(X_list)
    y = np.array(y_list, dtype=int)

    # Shuffle
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


# Convenience mapping
DATASETS = {
    "spirals": make_spirals,
    "moons": make_moons,
    "circles": make_circles,
    "xor": make_xor,
    "gaussian_blobs": make_gaussian_blobs,
    "mnist_subset": make_mnist_subset,
}
