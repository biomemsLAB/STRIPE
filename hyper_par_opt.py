# requirements
"""
matplotlib==3.5.2
numpy==1.21.5
pandas==1.4.4
scikit_learn==1.2.2
seaborn==0.11.2
torch==2.0.0
torchinfo==1.7.2
torchvision==0.15.1
ray==2.4.0
tabulate==0.9.0
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler



def load_data(data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/.data.lock")):
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform)

        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset

class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

from torch import Tensor
class DenseModel(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dtype=None) -> None:
        super(DenseModel, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.bias = bias
        self.device = device

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.selu = nn.SELU()

    def forward(self, input_tensor: Tensor) -> Tensor:
        # flatten_tensor = self.flatten(input_tensor)
        fc1_out = self.selu(self.fc1(input_tensor))
        fc2_out = self.selu(self.fc2(fc1_out))
        fc3_out = self.selu(self.fc3(fc2_out))
        return fc3_out

def calculate_metrics(true_labels, predicted_labels):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
    #print(cm)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    F1_Score = 2 * (PPV * TPR) / (PPV + TPR)
    # Overall accuracy
    ACC = TP / (TP + FP + FN)

    return cm, F1_Score, ACC

def train_cifar(config):
    net = Net(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    # To restore a checkpoint, use `session.get_checkpoint()`.
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
           model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    data_dir = os.path.abspath("./data")
    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        predicted_labels = []
        true_labels = []
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                predicted_labels += predicted.cpu().numpy().tolist()
                true_labels += labels.cpu().numpy().tolist()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        cm, f1_score, acc = calculate_metrics(true_labels, predicted_labels)
        print(cm)
        print(f1_score)
        print(f1_score[1])
        print(acc)

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `session.get_checkpoint()`
        # API in future iterations.
        os.makedirs("my_model", exist_ok=True)
        torch.save(
            (net.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("my_model")
        session.report({"loss": (val_loss / val_steps), "accuracy": correct / total, "f1_score": f1_score[1]}, checkpoint=checkpoint)
    print("Finished Training")

def test_best_model_cifar(best_result):
    best_trained_model = Net(best_result.config["l1"], best_result.config["l2"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = best_trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predicted_labels += predicted.cpu().numpy().tolist()
            true_labels += labels.cpu().numpy().tolist()

    cm, f1_score, acc = calculate_metrics(true_labels, predicted_labels)
    print(cm)
    print(f1_score)
    print(f1_score[1])
    print(acc)

    print("Best trial test set accuracy: {}".format(correct / total))
    print("Best trial test set f1_score: {}".format(f1_score[1]))


# here starts custom code for custom train and test loaders

def load_frame_from_disk(path_source):
    import numpy as np
    print('started loading frame from disk')
    frame = np.load(path_source, allow_pickle=True)
    print('frame loaded from disk')
    return frame


from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class WindowedFrameDataset(Dataset):
    """
    A PyTorch dataset that represents windowed frames of data and their labels.

    Parameters
    ----------
    data : numpy.ndarray or list
        The input data array of shape (num_frames, num_channels, height, width).
    labels : numpy.ndarray or list
        The target labels array of shape (num_frames,).

    Attributes
    ----------
    data : numpy.ndarray or list
        The input data array.
    transform : torchvision.transforms
        The transform applied to the input data.
    labels : numpy.ndarray or list
        The target labels array.

    Methods
    -------
    __len__()
        Returns the length of the dataset.
    __getitem__(idx)
        Returns the data and label of the specified index.

    Notes
    -----
    This class takes a numpy array or a list of windowed frames of data and their labels, and transforms them into a PyTorch dataset.
    It also applies a transform to the input data to convert it to a tensor.

    Example
    -------
    # create a dataset
    time_rows = np.random.randn(100, 3, 32, 32)
    labels = np.random.randint(0, 2, size=(100,))
    windowed_frame_dataset = WindowedFrameDataset(time_rows, labels)

    # create a dataloader
    batch_size = 32
    windowed_frame_dataloader = DataLoader(windowed_frame_dataset, batch_size=batch_size, shuffle=True)
    """

    def __init__(self, data, labels):
        self.data = data.astype('float32')
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.labels = labels.astype('float32')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]
        return data, labels


def create_dataloader(frame, batch_size=32):
    from torch.utils.data import DataLoader
    time_rows = frame['signals']  #frame[:, 0]
    labels = frame['label_per_window'] #frame[:, 1]
    windowed_frame_dataset = WindowedFrameDataset(time_rows, labels)
    windowed_frame_dataloader = DataLoader(windowed_frame_dataset, batch_size=batch_size, shuffle=True)
    return windowed_frame_dataloader


def create_dataloader_simple(data, labels, batch_size=32):
    from torch.utils.data import DataLoader
    time_rows = data
    windowed_frame_dataset = WindowedFrameDataset(time_rows, labels)
    windowed_frame_dataloader = DataLoader(windowed_frame_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return windowed_frame_dataloader


def train_custom(config):
    #net = Net(config["l1"], config["l2"])
    net = DenseModel(in_features=20, hidden_features=config["l1"], out_features=2)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    # To restore a checkpoint, use `session.get_checkpoint()`.
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
           model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    data_train_balanced = load_frame_from_disk('/mnt/MainNAS/BioMemsLaborNAS/Projekt_Ordner/STRIPE/debug/data/prepared_for_training/frames_x_train_res.npy')
    label_train_balanced = load_frame_from_disk('/mnt/MainNAS/BioMemsLaborNAS/Projekt_Ordner/STRIPE/debug/data/prepared_for_training/frames_y_train_res.npy')
    data_val = load_frame_from_disk('/mnt/MainNAS/BioMemsLaborNAS/Projekt_Ordner/STRIPE/debug/data/prepared_for_training/frames_x_val_crp.npy')
    label_val = load_frame_from_disk('/mnt/MainNAS/BioMemsLaborNAS/Projekt_Ordner/STRIPE/debug/data/prepared_for_training/frames_y_val_crp.npy')
    trainloader = create_dataloader_simple(data_train_balanced, label_train_balanced, batch_size=config["batch_size"])
    valloader = create_dataloader_simple(data_val, label_val, batch_size=config["batch_size"])

    """
    data_dir = os.path.abspath("./data")
    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    """

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.type(torch.LongTensor).to(device) # inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        predicted_labels = []
        true_labels = []
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.type(torch.LongTensor).to(device) # labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                predicted_labels += predicted.cpu().numpy().tolist()
                true_labels += labels.cpu().numpy().tolist()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        cm, f1_score, acc = calculate_metrics(true_labels, predicted_labels)
        print(cm)
        print(f1_score)
        print(f1_score[1])
        print(acc)

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `session.get_checkpoint()`
        # API in future iterations.
        os.makedirs("my_model", exist_ok=True)
        torch.save(
            (net.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("my_model")
        session.report({"loss": (val_loss / val_steps), "accuracy": correct / total, "f1_score": f1_score[1]}, checkpoint=checkpoint)
    print("Finished Training")

def test_best_model_custom(best_result):
    # best_trained_model = Net(best_result.config["l1"], best_result.config["l2"])
    best_trained_model = DenseModel(in_features=10, hidden_features=best_result.config["l1"], out_features=2)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)


    data_test = load_frame_from_disk('/mnt/MainNAS/BioMemsLaborNAS/Projekt_Ordner/STRIPE/debug/data/prepared_for_training/frames_x_test_crp.npy')
    label_test = load_frame_from_disk('/mnt/MainNAS/BioMemsLaborNAS/Projekt_Ordner/STRIPE/debug/data/prepared_for_training/frames_y_test_crp.npy')
    testloader = create_dataloader_simple(data_test, label_test, batch_size=best_result.config["batch_size"])


    """
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)
    """

    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.type(torch.LongTensor).to(device) # labels.to(device)
            outputs = best_trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predicted_labels += predicted.cpu().numpy().tolist()
            true_labels += labels.cpu().numpy().tolist()

    cm, f1_score, acc = calculate_metrics(true_labels, predicted_labels)
    print(cm)
    print(f1_score)
    print(f1_score[1])
    print(acc)

    print("Best trial test set accuracy: {}".format(correct / total))
    print("Best trial test set f1_score: {}".format(f1_score[1]))



def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    config = {
        "l1": tune.choice([40, 50, 60]),
        #"l1": tune.sample_from(lambda _: 2 ** np.random.randint(10, 100)),
        #"l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([2, 4, 16, 32])
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_custom), # train_cifar
            resources={"cpu": 2, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss", # "loss"
            mode="min", # "min"
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result("loss", "min") # "loss", "min"

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))
    print("Best trial final f1_score: {}".format(
        best_result.metrics["f1_score"]))

    #test_best_model_cifar(best_result)
    test_best_model_custom(best_result)

if __name__ == "__main__":


    print('hih')
    main(num_samples=4, max_num_epochs=4, gpus_per_trial=2)
    print('finish')