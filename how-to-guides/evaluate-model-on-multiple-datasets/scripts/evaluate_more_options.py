import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class PretrainedModel(pl.LightningModule):
    # ...

    def test_step(self, batch, batch_idx):
        # ...
        # Log images
        if batch_idx == 0:
            self.log_images(inputs, targets, preds)

    def test_epoch_end(self, outputs):
        # ...
        # Log confusion matrix
        self.log_confusion_matrix(targets_all, preds_all)

    def log_images(self, inputs, targets, preds):
        # Implement code to log images to Neptune here
        pass

    def log_confusion_matrix(self, targets, preds):
        # Compute the confusion matrix
        cm = confusion_matrix(targets, preds)

        # Plot the confusion matrix
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

        # Log the confusion matrix to Neptune
        run["confusion_matrix"].upload(neptune.types.File.as_image(fig))
