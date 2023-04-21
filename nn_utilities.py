import numpy as np
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
            if verbose:
                print(f"Early Stop: Epoch Counter at {self.counter} with Patience {self.patience}, current min loss {self.min_validation_loss}/{self.min_validation_loss + self.min_delta}")
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if verbose:
                print(f"Early Stop: Epoch Counter at {self.counter} with Patience {self.patience}, current min loss {self.min_validation_loss}/{self.min_validation_loss + self.min_delta}")
            if self.counter >= self.patience:
                if verbose:
                    print("Stopping early to prevent overfitting")
                return True
        return False

import torch
import torch.nn as nn
def compare_models_acc_over_epoch(train_dataloader, eval_dataloader, test_dataloader, *models: nn.Module, epochs=100, learning_rate=0.001, path_to_save="") -> None:
    """
    Compare the train and eval accuracy over epochs for multiple PyTorch models.

    Args:
    - train_dataloader: PyTorch DataLoader for training dataset
    - eval_dataloader: PyTorch DataLoader for validation dataset
    - test_dataloader: PyTorch DataLoader for testing dataset
    - *models: one or more PyTorch models to compare

    Returns: None

    """
    from nn_handle import handle_model
    acc_list_per_noise_level = []
    model_handlers = []
    model_names = []


    for model in models:
        model_handlers.append(handle_model(model, train_dataloader, eval_dataloader, test_dataloader))
        model_names.append(model.__class__.__name__)
        #models[0].__class__.__name__
    train_acc_list = []
    eval_acc_list = []
    train_loss_list = []

    epochs = epochs
    learning_rate = learning_rate
    for model_runner in model_handlers:
        model_runner.run(epochs=epochs, learning_rate=learning_rate)
        train_acc_list.append(model_runner.train_acc)
        eval_acc_list.append(model_runner.eval_acc)
        train_loss_list.append(model_runner.train_loss)

    path = path_to_save + "/Compare"


    import pandas as pd
    df_train_acc = pd.DataFrame(train_acc_list).T
    test_path_save = path + "_train_acc"
    titel = "Train Accuracy comparison of NN"
    plot_acc_df(df_train_acc, model_names, test_path_save, titel)

    df_eval_acc = pd.DataFrame(eval_acc_list).T
    eval_path_save = path + "_eval_acc"
    titel = "Eval Accuracy comparison of NN"
    plot_acc_df(df_eval_acc, model_names, eval_path_save, titel)

    df_train_loss = pd.DataFrame(train_loss_list).T
    train_loss_path_save = path + "_train_loss"
    titel = "Loss comparison of NN"
    plot_loss_df(df_train_loss, model_names, train_loss_path_save, titel)





def plot_acc_df(df, model_names, path_save, titel):
    """
    Plot the accuracy comparison chart for PyTorch models.

    Args:
    - df: Pandas dataframe containing accuracy values
    - model_names: list of model names
    - path_save: path to save the plot
    - titel: plot title

    Returns: None

    """
    df.columns = model_names
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    for model_name in model_names:
        path_save = path_save + "_" + model_name
    # path_save = path_save + "_train_" + ".png"
    path_save = path_save + ".png"
    figure, axes = plt.subplots()
    epochs = len(df.index)
    axes.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if epochs < 10:
        axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
        axes.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.grid()
    plt.ylabel("Accuracy in %")
    plt.xlabel("Epochs")
    plt.title(titel)
    for i in range(df.shape[1]):
        sns.lineplot(data=df, x=df.index + 1, y=df.iloc[:, i], ax=axes, label=df.columns[i], marker="*",
                     markersize=8)

    plt.legend(loc='lower right')
    # axes.legend(labels=["Acc1", "Acc2"])
    plt.savefig(path_save)
    plt.close()

def plot_loss_df(df, model_names, path_save, titel):
    """
    Plot the loss comparison chart for PyTorch models.

    Args:
    - df: Pandas dataframe containing loss values
    - model_names: list of model names
    - path_save: path to save the plot
    - titel: plot title

    Returns: None

    """
    df.columns = model_names
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    for model_name in model_names:
        path_save = path_save + "_" + model_name
    # path_save = path_save + "_train_" + ".png"
    path_save = path_save + ".png"
    figure, axes = plt.subplots()
    epochs = len(df.index)
    if epochs < 10:
        axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
        axes.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.grid()
    plt.ylabel("Loss")
    plt.xlabel("Total Batches seen")
    plt.title(titel)
    for i in range(df.shape[1]):
        sns.lineplot(data=df, x=df.index + 1, y=df.iloc[:, i], ax=axes, label=df.columns[i])

    plt.legend(loc='upper right')
    # axes.legend(labels=["Acc1", "Acc2"])
    plt.savefig(path_save)
    plt.close()

import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
  r"""Implements Lion algorithm."""

  def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
    """Initialize the hyperparameters.
    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
      lr (float, optional): learning rate (default: 1e-4)
      betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.99))
      weight_decay (float, optional): weight decay coefficient (default: 0)
    """

    if not 0.0 <= lr:
      raise ValueError('Invalid learning rate: {}'.format(lr))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
    defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
    super().__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.
    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    Returns:
      the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        # Perform stepweight decay
        p.data.mul_(1 - group['lr'] * group['weight_decay'])

        grad = p.grad
        state = self.state[p]
        # State initialization
        if len(state) == 0:
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(p)

        exp_avg = state['exp_avg']
        beta1, beta2 = group['betas']

        # Weight update
        update = exp_avg * beta1 + grad * (1 - beta1)
        p.add_(torch.sign(update), alpha=-group['lr'])
        # Decay the momentum running average coefficient
        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

    return loss



