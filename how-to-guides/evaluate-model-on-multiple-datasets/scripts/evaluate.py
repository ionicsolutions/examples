# Model
import neptune
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import NeptuneLogger
from torchmetrics.classification import Accuracy
from torchvision import datasets as DataSets
from torchvision import models


# %%
class PretrainedModel(pl.LightningModule):
    def __init__(self):
        super(PretrainedModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)


# Datasets
PATH = "./data"
datasets = {
    "cifar-10": DataSets.CIFAR10(
        root=f"{PATH}/cifar-10", train=False, download=True, transform=None
    ),
    "cifar-100": DataSets.CIFAR100(
        root=f"{PATH}/data/cifar-100", train=False, download=True, transform=None
    ),
    "stl-10": DataSets.STL10(root=f"{PATH}/stl-10", split="test", download=True, transform=None),
}


for _, dataset in datasets.items():
    dataloader = torch.utils.data.Dataloader(dataset, batch_size=32, num_workers=4, shuffle=False)

    # Create a new run for each dataset
    # Initialize the NeptuneLogger
    neptune_logger = NeptuneLogger()

    # Log the dataset as an artifact
    neptune_logger.experiment[f"data/{dataset}"].track_files(dataset.root)

    # Log dataset samples
    neptune_logger.experiment[f"data/samples"] = len(dataset)

    # Log dataset classes names
    neptune_logger.experiment[f"data/num_classes"] = len(dataset.classes)

    # Log dataset class to index mapping
    neptune_logger.experiment[f"data/classes"] = dataset.class_to_idx

    # Initialize the pre-trained model with the appropriate number of classes
    model = PretrainedModel(num_classes=len(dataset.classes))

    # Create the PyTorch Lightning Trainer with the NeptuneLogger
    trainer = pl.Trainer(logger=neptune_logger, max_epochs=1)

    # Evaluate the model on the dataset
    trainer.test(model, dataloader=dataloader)

    # Log the model as an artifact
    neptune_logger.experiment["model/checkpoints}"].track_files(neptune.File.as_pickle(model))

    # End the run
    neptune_logger.experiment.stop()
