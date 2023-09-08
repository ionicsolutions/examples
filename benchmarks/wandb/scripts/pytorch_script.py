import time

import neptune
import numpy as np
import torch
from neptune.types import File
from neptune.utils import stringify_unsupported
from neptune_pytorch import NeptuneLogger
from torch import nn, optim
from torchvision import datasets, transforms


def main():
    # Hyperparams for training
    parameters = {
        "lr": 1e-2,
        "bs": 128,
        "input_sz": 32 * 32 * 3,
        "n_classes": 10,
        "model_filename": "basemodel",
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "epochs": 2,
    }

    # Model
    class Model(nn.Module):
        def __init__(self, input_sz, hidden_dim, n_classes):
            super(Model, self).__init__()
            self.seq_model = nn.Sequential(
                nn.Linear(input_sz, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, n_classes),
            )

        def forward(self, input):
            x = input.view(-1, 32 * 32 * 3)
            return self.seq_model(x)

    model = Model(parameters["input_sz"], parameters["input_sz"], parameters["n_classes"]).to(
        parameters["device"]
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=parameters["lr"])

    # Data
    data_dir = "data/CIFAR10"
    compressed_ds = "./data/CIFAR10/cifar-10-python.tar.gz"
    data_tfms = {
        "train": transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    trainset = datasets.CIFAR10(data_dir, transform=data_tfms["train"], download=True)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=parameters["bs"], shuffle=True, num_workers=0
    )
    validset = datasets.CIFAR10(data_dir, train=False, transform=data_tfms["train"], download=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=parameters["bs"], num_workers=0)

    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    for epoch in range(parameters["epochs"]):
        for i, (x, y) in enumerate(trainloader, 0):
            x, y = x.to(parameters["device"]), y.to(parameters["device"])
            optimizer.zero_grad()
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, y)
            acc = (torch.sum(preds == y.data)) / len(x)

            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    num_trials = 50  # number of trials
    times = []  # list to store times
    for i in range(num_trials):
        start_time = time.time()
        main()
        end_time = time.time()
        times.append(end_time - start_time)

    # Calculate the average time
    average_time = np.mean(times)

    # Calculate the standard deviation
    std_dev = np.std(times)

    print(
        f"Average execution time over {num_trials} trials: {average_time} seconds with standard deviation of {std_dev} seconds"
    )
