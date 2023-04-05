class EarlyStopper:
    """Helper class for early stopping during training.

    Early stopping is a technique used to prevent overfitting in machine learning models.
    It works by stopping the training process early, before the model has a chance to overfit
    to the training data. Early stopping is based on monitoring the validation loss during
    training. If the validation loss stops improving or starts getting worse, training is
    stopped to prevent overfitting.

    Parameters:
    -----------
    patience : int, default=1
        Number of epochs to wait for improvement in validation loss before stopping training.
    min_delta : float, default=0
        Minimum change in validation loss to be considered as an improvement.

    Attributes:
    -----------
    counter : int
        Counter that keeps track of the number of epochs without improvement in validation loss.
    min_validation_loss : float
        Minimum validation loss observed during training.

    Methods:
    --------
    early_stop(validation_loss, verbose=True)
        Check if early stopping criterion is met based on validation loss.
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss, verbose=True):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                if verbose:
                    print("Stopping Early to prevent overfitting :)")
                return True
        return False