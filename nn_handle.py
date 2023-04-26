import torch
import torch.nn as nn
from torchinfo import summary
from nn_utilities import EarlyStopper
from alive_progress import alive_bar

class handle_model():
    """
    A class for handling PyTorch model training and testing on provided datasets.

    Args:
    -----------
    model: A PyTorch model object to train and test.
    train_dataloader: A PyTorch DataLoader object containing training data.
    test_dataloader: A PyTorch DataLoader object containing testing data.

    Attributes:
    -----------
    model: A PyTorch model object to train and test.
    train_dataloader: A PyTorch DataLoader object containing training data.
    test_dataloader: A PyTorch DataLoader object containing testing data.
    epochs: An integer representing the number of epochs to train the model.

    Methods:
    -----------
    run(): Runs the model training and testing on provided datasets.
    train(dataloader, model, loss_fn, optimizer, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')): Trains the model on the provided training dataset.
    test(dataloader, model, loss_fn, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')): Tests the trained model on the provided testing dataset.
    evaluate(dataloader, model, loss_fn, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')): Evaluates the trained model on the provided validation dataset.
    plot_training_acc(): Plots the training accuracy of the model with respect to the number of epochs.

    """
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.eval_dataloader = eval_dataloader
        self.model = model
        self.model_info = ...
        self.train_acc = []
        self.train_acc_with_epoch = []
        self.eval_acc = []
        self.eval_acc_with_epoch = []
        self.test_acc = []
        self.test_acc_with_epoch = []
        # self.training_loss = []
        self.current_acc = 0

        self.train_loss = []
        self.train_loss_with_epoch = []

        self.avg_train_loss = []
        self.avg_train_loss_with_epoch = []
        self.avg_eval_loss = []
        self.avg_eval_loss_with_epoch = []
        self.final_avg_test_loss = []
        self.final_avg_test_loss_with_epoch = []


        self.best_train_acc = 0
        self.best_eval_acc = 0
        self.best_test_acc = 0

        self.avg_tet_acc = 0

        self.name_of_model = model.__class__.__name__
        self.device = device
        self.path_to_save = ""
        self.flag = True

    def run(self, epochs=5, learning_rate=1e-3, loss_fn=nn.CrossEntropyLoss(), path_to_save=""):
        """
        Trains and tests the model on provided datasets for specified number of epochs.
        """
        self.batch_size = self.train_dataloader.batch_size
        self.model.to(self.device)
        self.epochs = epochs
        self.learing_rate = learning_rate
        self.loss_fn = loss_fn
        self.path_to_save = path_to_save
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        from nn_utilities import Lion
        self.optimizer = Lion(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=4, verbose=False)
        self.early_stopper = EarlyStopper(patience=8, min_delta=0.008)
        try:
            self.model_info = summary(self.model, input_size=(self.batch_size, 10))
        except:
            print("Summary not working")
        self.check_path()
        print(f"Running model")
        for self.epoch in range(self.epochs):
            print(f"Epoch {self.epoch + 1} of {self.name_of_model}\n-------------------------------")
            self.train(self.train_dataloader)
            # self.scheduler.step(loss)
            self.evaluate(self.eval_dataloader)
            if self.early_stopper.early_stop(self.eval_loss):
                break
            if self.flag or self.current_acc > max(self.train_acc):
                print("Model saved")
                self.save_model()
                self.flag = False
        print("\n")
        self.test(self.test_dataloader)
        self.best_train_acc = max(self.train_acc)
        self.best_eval_acc = max(self.eval_acc)
        self.best_test_acc = max(self.test_acc)
        from statistics import mean
        self.avg_tet_acc = mean([self.best_train_acc, self.best_eval_acc, self.best_test_acc])
        path_json = path_to_save + "/" + str(self.name_of_model) + ".json"
        self.save_json(path_json)


        print("Done!")
    def train(self, dataloader):
        """
        Trains the provided PyTorch model on the provided training dataset.

        Args:
        dataloader (DataLoader): A PyTorch DataLoader object containing training data.

        Returns:
        None
        """

        n_total_steps = len(dataloader)
        self.model.train()
        temp_train_loss = []
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.type(torch.LongTensor).to(self.device)

            # Compute prediction error
            pred = self.model(X)
            #max_indices = pred.argmax(dim=1)
            #max_tensor = max_indices

            # y = y.squeeze(1)
            loss = self.loss_fn(pred, y)
            self.train_loss.append(loss.item())
            # self.scheduler.step(loss)
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            verbose_step = int(n_total_steps/5)
            if (batch + 1) % verbose_step == 0:
                print(f"Epoch [{self.epoch + 1}/{ self.epochs}], Step [{batch + 1}/{n_total_steps}], Loss: {loss.item():.4f}, Learning Rate {self.optimizer.param_groups[0]['lr']}")
                #print(self.model.sl1.weight)
                #print(model.sl2.weight)
                #print(model.sl2.connections)

        self.train_loss_with_epoch.append([self.epoch, self.train_loss])
        # self.scheduler.step(loss)
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        check_train_loss, correct = 0, 0
        with torch.no_grad():
            with alive_bar(total=len(dataloader), force_tty=True, title="Calculating Train accuracy:\t\t") as bar:
                for X, y in dataloader:
                    X, y = X.to(self.device), y.type(torch.LongTensor).to(self.device)
                    # y = y.squeeze(1)
                    pred = self.model(X)
                    check_train_loss += self.loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                    bar()
        #self.scheduler.step(self.train_loss)
        check_train_loss /= num_batches
        correct /= size
        print(f"Train Error: \t\tAccuracy: {(100 * correct):>0.2f}%, average train loss: \t\t{check_train_loss:>8f}")
        self.train_acc.append(100 * correct)
        self.current_acc = 100 * correct
        self.train_acc_with_epoch.append([self.epoch + 1, 100 * correct])
        self.avg_train_loss.append(check_train_loss)
        self.avg_train_loss_with_epoch.append([self.epoch, check_train_loss])

    def evaluate(self, dataloader):
        """
        Evaluates the provided PyTorch model on the provided validation dataset.

        Args:
        dataloader: A PyTorch DataLoader object containing validation data.

        Returns:
        None
        """
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        self.eval_loss, correct = 0, 0
        with torch.no_grad():
            with alive_bar(total=len(dataloader), force_tty=True, title="Calculating Eval accuracy:\t\t") as bar:
                for X, y in dataloader:
                    X, y = X.to(self.device), y.type(torch.LongTensor).to(self.device)
                    # y = y.squeeze(1)
                    pred = self.model(X)
                    self.eval_loss += self.loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                    bar()
        self.scheduler.step(self.eval_loss)
        self.eval_loss /= num_batches
        correct /= size
        print(f"Evaluation Error: \tAccuracy: {(100*correct):>0.2f}%, average evaluation loss: \t{self.eval_loss:>8f}")
        self.eval_acc.append(100*correct)
        self.eval_acc_with_epoch.append([self.epoch+1, 100*correct])
        self.avg_eval_loss.append(self.eval_loss)
        self.avg_eval_loss_with_epoch.append([self.epoch, self.eval_loss])

    def test(self, dataloader):
        """
        Tests the trained model on the provided test dataset.

        Args:
        dataloader: A PyTorch DataLoader object containing test data.

        Returns:
        None
        """
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            print("Calculation Test Accuracy and average Loss")
            with alive_bar(total=len(dataloader), force_tty=True, title="Calculating Test accuracy:\t\t") as bar:
                for X, y in dataloader:
                    X, y = X.to(self.device), y.type(torch.LongTensor).to(self.device)
                    # y = y.squeeze(1)
                    pred = self.model(X)
                    test_loss += self.loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                    bar()
        test_loss /= num_batches
        self.test_loss = test_loss
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        self.test_acc.append(100*correct)
        self.test_acc_with_epoch.append([self.epoch+1, 100*correct])
        self.final_avg_test_loss.append(self.test_loss)
        self.final_avg_test_loss_with_epoch.append([self.epoch, self.test_loss])

    def plot_training_acc(self, save_string="training_acc_plot.png"):
        """
        This method plots a line graph of the accuracy of the neural network during training over epochs and saves it to a file. The graph shows the accuracy of the network at each epoch, using the evaluation accuracy data stored in the instance variable 'eval_acc_with_epoch'. The x-axis represents the epoch number and the y-axis represents the accuracy in percentage.

        Parameters:

        save_string (string): The filename to save the plot as. Default is "training_acc_plot.png".
        Returns:

        None. The plot is saved as a file.
        """

        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        figure, axes = plt.subplots()
        if self.epochs < 100:
            axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
            axes.xaxis.set_major_formatter(ticker.ScalarFormatter())
        plt.grid()
        plt.ylabel("Accuracy in %")
        plt.xlabel("Epochs")
        plt.title("Accuracy comparison of NN with MNIST")
        df = pd.DataFrame(data=self.eval_acc_with_epoch, columns=["Epoch", "Accuracy"])
        sns.lineplot(data=df, x=df.Epoch, y=df.Accuracy, ax=axes)
        plt.savefig(save_string)
        plt.close()

    def save_json(self, filename):
        """
        Speichert alle wichtigen self-Parameter in einer JSON-Datei.

        Args:
        filename (str): Name der Datei, in der die Daten gespeichert werden sollen.

        Returns:
        None
        """
        import json
        with open(filename, 'w') as f:
            json.dump({
                # 'train_dataloader': self.train_dataloader.dataset,
                # 'test_dataloader': self.test_dataloader.dataset,
                # 'eval_dataloader': self.eval_dataloader.dataset,
                'model': str(self.model),
                'model_info': str(self.model_info),
                'train_acc': self.train_acc,
                'train_acc_with_epoch': self.train_acc_with_epoch,
                'eval_acc': self.eval_acc,
                'eval_acc_with_epoch': self.eval_acc_with_epoch,
                'test_acc': self.test_acc,
                'test_acc_with_epoch': self.test_acc_with_epoch,
                'train_loss': self.train_loss,
                'train_loss_with_epoch': self.train_loss_with_epoch,
                'avg_train_loss': self.avg_train_loss,
                'avg_train_loss_with_epoch': self.avg_train_loss_with_epoch,
                'avg_eval_loss': self.avg_eval_loss,
                'avg_eval_loss_with_epoch': self.avg_eval_loss_with_epoch,
                'final_avg_test_loss': self.final_avg_test_loss,
                'final_avg_test_loss_with_epoch': self.final_avg_test_loss_with_epoch,
                'best_train_acc': self.best_train_acc,
                'best_eval_acc': self.best_eval_acc,
                'best_test_acc': self.best_test_acc,
                'avg_tet_acc': self.avg_tet_acc,
                'name_of_model': self.name_of_model,
                'device': str(self.device),
                'epochs': self.epochs,
                'learing_rate': self.learing_rate,
                'loss_fn': str(self.loss_fn),
                'optimizer': str(self.optimizer),
                'scheduler': str(self.scheduler),
                'early_stopper': str(self.early_stopper)
            }, f, indent=4)

    def check_path(self):
        import os
        if not os.path.isdir(self.path_to_save):
            os.mkdir(self.path_to_save)

    def save_model(self):
        model_path = self.path_to_save + "/" + self.name_of_model + ".pth"
        torch.save(self.model, model_path)



