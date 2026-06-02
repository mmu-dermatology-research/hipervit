"""
Early stopping utility.

Reference: https://github.com/jeffheaton/app_deep_learning/blob/main/t81_558_class_03_4_early_stop.ipynb
"""

import copy


class EarlyStopping:
    """Stop training when a monitored validation metric stops improving.

    Calls to ``__call__`` return ``True`` the first time the patience budget
    is exhausted, and optionally restore the best model weights in-place.

    Args:
        patience (int): Number of epochs with no improvement after which
            training will be stopped.
        min_delta (float): Minimum absolute change that qualifies as an
            improvement.  Keep this small; a very small default means any
            genuine reduction counts.
        restore_best_weights (bool): If ``True``, the model's ``state_dict``
            is restored to the best-seen weights when stopping is triggered.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-7,
        restore_best_weights: bool = True,
    ):
        self.patience             = patience
        self.min_delta            = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_model = None
        self.best_loss  = None
        self.counter    = 0
        self.status     = ""

    def __call__(self, model, val_loss: float) -> bool:
        """Evaluate the current validation loss and decide whether to stop.

        Pass a *negated* metric (e.g. ``-auc``) to maximise instead of
        minimise.

        Args:
            model (nn.Module): The model being trained.  Its weights are
                snapshotted whenever a new best is found, and restored when
                stopping is triggered (if ``restore_best_weights=True``).
            val_loss (float): Current validation loss (lower is better).

        Returns:
            bool: ``True`` if training should stop, ``False`` otherwise.
        """
        if self.best_loss is None:
            self.best_loss  = val_loss
            self.best_model = copy.deepcopy(model.state_dict())

        elif self.best_loss - val_loss >= self.min_delta:
            self.best_loss  = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter    = 0
            self.status     = f"Improvement found, counter reset to {self.counter}"

        else:
            self.counter += 1
            self.status   = f"No improvement in the last {self.counter} epochs"

            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True

        return False
