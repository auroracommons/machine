"""
Sequential neural network model with training loop, callbacks, and metrics.

The Network class assembles layers into a sequential model and provides
a high-level API for training with batch processing, metric tracking,
and callback support.
"""

import numpy as np
import time
from .layers import Layer, Dense, Dropout, BatchNorm
from .losses import get_loss
from .optimizers import OPTIMIZERS, LRScheduler


class TrainingHistory:
    """Records training metrics across epochs."""

    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.learning_rates = []
        self.epoch_times = []
        self.gradient_norms = []

    def record(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                getattr(self, key).append(value)


class Callback:
    """Base callback class."""

    def on_epoch_begin(self, epoch, history):
        pass

    def on_epoch_end(self, epoch, history):
        pass

    def on_train_begin(self, history):
        pass

    def on_train_end(self, history):
        pass


class PrintProgress(Callback):
    """Prints training progress."""

    def __init__(self, print_every=10):
        self.print_every = print_every

    def on_epoch_end(self, epoch, history):
        if (epoch + 1) % self.print_every == 0 or epoch == 0:
            msg = f"  Epoch {epoch + 1:4d} | Loss: {history.train_loss[-1]:.6f}"
            if history.train_acc:
                msg += f" | Acc: {history.train_acc[-1]:.4f}"
            if history.val_loss:
                msg += f" | Val Loss: {history.val_loss[-1]:.6f}"
            if history.learning_rates:
                msg += f" | LR: {history.learning_rates[-1]:.6f}"
            print(msg)


class EarlyStopping(Callback):
    """Stop training when validation loss stops improving."""

    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait = 0
        self.stopped = False

    def on_epoch_end(self, epoch, history):
        if history.val_loss:
            current = history.val_loss[-1]
        else:
            current = history.train_loss[-1]

        if current < self.best_loss - self.min_delta:
            self.best_loss = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped = True


class Network:
    """
    Sequential neural network model.

    Build a network by adding layers sequentially, then train with
    the fit() method.

    Example:
        net = Network()
        net.add(Dense(2, 64, activation='relu'))
        net.add(Dense(64, 32, activation='relu'))
        net.add(Dense(32, 2))
        net.compile(optimizer='adam', loss='cross_entropy', lr=0.01)
        history = net.fit(X_train, y_train, epochs=100, batch_size=32)
    """

    def __init__(self):
        self.layers = []
        self.optimizer = None
        self.loss_fn = None
        self.scheduler = None
        self._compiled = False

    def add(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)
        return self

    def compile(self, optimizer="adam", loss="cross_entropy", lr=0.01,
                scheduler=None, scheduler_kwargs=None, grad_clip=None,
                **optimizer_kwargs):
        """
        Configure the network for training.

        Args:
            optimizer: Name of optimizer or Optimizer instance
            loss: Name of loss function or Loss instance
            lr: Learning rate
            scheduler: Name of LR scheduler or None
            scheduler_kwargs: Additional kwargs for the scheduler
            grad_clip: Max gradient norm for clipping (None = no clipping)
            **optimizer_kwargs: Additional kwargs for the optimizer
        """
        # Collect all parameters
        all_params = []
        for layer in self.layers:
            all_params.extend(layer.parameters())

        # Build optimizer
        if isinstance(optimizer, str):
            optimizer_cls = OPTIMIZERS[optimizer.lower()]
            self.optimizer = optimizer_cls(all_params, lr=lr, grad_clip=grad_clip, **optimizer_kwargs)
        else:
            self.optimizer = optimizer

        # Build loss
        self.loss_fn = get_loss(loss)

        # Build scheduler
        if scheduler is not None:
            from .optimizers import SCHEDULERS
            scheduler_kwargs = scheduler_kwargs or {}
            self.scheduler = SCHEDULERS[scheduler](self.optimizer, **scheduler_kwargs)

        self._compiled = True

    def forward(self, x):
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, x):
        """Run forward pass in eval mode."""
        self._set_mode(training=False)
        out = self.forward(x)
        self._set_mode(training=True)
        return out

    def _set_mode(self, training=True):
        for layer in self.layers:
            if training:
                layer.train()
            else:
                layer.eval()

    def _compute_accuracy(self, predictions, targets):
        """Compute classification accuracy."""
        if predictions.ndim == 1 or predictions.shape[-1] == 1:
            # Binary classification
            preds_flat = predictions.ravel()
            tgts_flat = targets.ravel()
            pred_labels = (preds_flat > 0.5).astype(int)
            return np.mean(pred_labels == tgts_flat)
        else:
            # Multi-class classification
            pred_labels = np.argmax(predictions, axis=-1)
            if targets.ndim > 1:
                true_labels = np.argmax(targets, axis=-1)
            else:
                true_labels = targets.astype(int)
            return np.mean(pred_labels == true_labels)

    def _compute_gradient_norm(self):
        """Compute global gradient norm across all parameters."""
        total_norm = 0.0
        for layer in self.layers:
            for param in layer.parameters():
                total_norm += np.sum(param.grad ** 2)
        return np.sqrt(total_norm)

    def fit(self, X, y, epochs=100, batch_size=32, validation_data=None,
            callbacks=None, shuffle=True, verbose=True, compute_accuracy=True):
        """
        Train the network.

        Args:
            X: Training input data, shape (N, features)
            y: Training targets
            epochs: Number of training epochs
            batch_size: Mini-batch size (None = full batch)
            validation_data: Tuple of (X_val, y_val) or None
            callbacks: List of Callback instances
            shuffle: Whether to shuffle data each epoch
            verbose: Whether to print a progress summary
            compute_accuracy: Whether to track accuracy

        Returns:
            TrainingHistory with recorded metrics
        """
        if not self._compiled:
            raise RuntimeError("Call compile() before fit()")

        history = TrainingHistory()
        callbacks = callbacks or []

        if verbose and not any(isinstance(c, PrintProgress) for c in callbacks):
            print_every = max(1, epochs // 20)
            callbacks.append(PrintProgress(print_every=print_every))

        N = X.shape[0]
        if batch_size is None:
            batch_size = N

        # Notify callbacks
        for cb in callbacks:
            cb.on_train_begin(history)

        for epoch in range(epochs):
            t0 = time.time()

            for cb in callbacks:
                cb.on_epoch_begin(epoch, history)

            # Shuffle
            if shuffle:
                idx = np.random.permutation(N)
                X_shuffled = X[idx]
                y_shuffled = y[idx]
            else:
                X_shuffled = X
                y_shuffled = y

            # Mini-batch training
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            n_batches = 0

            self._set_mode(training=True)

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                batch_n = end - start

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                predictions = self.forward(X_batch)

                # Compute loss
                loss = self.loss_fn(predictions, y_batch)
                epoch_loss += loss * batch_n

                # Backward pass
                grad = self.loss_fn.backward()
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)

                # Update parameters
                self.optimizer.step()

                if compute_accuracy:
                    epoch_correct += self._compute_accuracy(predictions, y_batch) * batch_n
                epoch_total += batch_n
                n_batches += 1

            # Record epoch metrics
            avg_loss = epoch_loss / N
            history.record(train_loss=avg_loss)

            if compute_accuracy:
                history.record(train_acc=epoch_correct / epoch_total)

            # Gradient norm (from last batch)
            grad_norm = self._compute_gradient_norm()
            history.record(gradient_norms=grad_norm)

            # Learning rate
            history.record(learning_rates=self.optimizer.lr)

            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                self._set_mode(training=False)
                val_preds = self.forward(X_val)
                val_loss = self.loss_fn(val_preds, y_val)
                history.record(val_loss=val_loss)
                if compute_accuracy:
                    history.record(val_acc=self._compute_accuracy(val_preds, y_val))
                self._set_mode(training=True)

            # Epoch time
            history.record(epoch_times=time.time() - t0)

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Callbacks
            for cb in callbacks:
                cb.on_epoch_end(epoch, history)

            # Check early stopping
            for cb in callbacks:
                if isinstance(cb, EarlyStopping) and cb.stopped:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    for cb2 in callbacks:
                        cb2.on_train_end(history)
                    return history

        for cb in callbacks:
            cb.on_train_end(history)

        return history

    def get_layer_weights(self):
        """Return a list of (weight_data, bias_data) for each Dense layer."""
        weights = []
        for layer in self.layers:
            if isinstance(layer, Dense):
                w = layer.weight.data.copy()
                b = layer.bias.data.copy() if layer.bias is not None else None
                weights.append((w, b))
        return weights

    def get_layer_gradients(self):
        """Return a list of (weight_grad, bias_grad) for each Dense layer."""
        grads = []
        for layer in self.layers:
            if isinstance(layer, Dense):
                wg = layer.weight.grad.copy()
                bg = layer.bias.grad.copy() if layer.bias is not None else None
                grads.append((wg, bg))
        return grads

    def count_parameters(self):
        """Count total trainable parameters."""
        total = 0
        for layer in self.layers:
            for param in layer.parameters():
                total += param.data.size
        return total

    def summary(self):
        """Print a summary of the network architecture."""
        print("=" * 60)
        print(f"{'Layer':<25} {'Output Shape':<20} {'# Params':<10}")
        print("-" * 60)
        total_params = 0
        for i, layer in enumerate(self.layers):
            name = layer.__class__.__name__
            if isinstance(layer, Dense):
                shape = f"(*, {layer.out_features})"
                n_params = layer.in_features * layer.out_features
                if layer.use_bias:
                    n_params += layer.out_features
                if layer.activation:
                    name += f"({layer.activation.__class__.__name__})"
            elif isinstance(layer, BatchNorm):
                shape = f"(*, {layer.num_features})"
                n_params = 2 * layer.num_features
            elif isinstance(layer, Dropout):
                shape = "same"
                n_params = 0
                name += f"(p={layer.p})"
            else:
                shape = "same"
                n_params = 0

            total_params += n_params
            print(f"  {name:<23} {shape:<20} {n_params:<10,}")
        print("-" * 60)
        print(f"  Total parameters: {total_params:,}")
        print("=" * 60)
