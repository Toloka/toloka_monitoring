import torch
from torch import nn
from torchvision import models
import pytorch_lightning as pl
from torchmetrics.functional.classification import accuracy


class ImageClassifier(pl.LightningModule):
    def __init__(self,
                 **kwargs):
        super().__init__()
        self.trunk = models.resnet18(pretrained=True)
        self.head = nn.Linear(1000, 2)
        self.loss = nn.CrossEntropyLoss()

        for name, param in self.trunk.named_parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        return {
            'optimizer': optimizer,
        }

    def forward(self, x):
        embedding = self.trunk(x)
        return self.head(embedding)

    def predict_proba(self, x):
        return torch.sigmoid(self.head(self.trunk(x)))

    def _process_batch(self, batch, batch_idx, **kwargs):
        images, labels = batch
        logits = self(images)
        loss = self.loss(logits, labels)
        return logits, loss

    def training_step(self, batch, batch_idx):
        logits, loss = self._process_batch(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits, loss = self._process_batch(batch, batch_idx)
        acc = accuracy(logits, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits, loss = self._process_batch(batch, batch_idx)
        acc = accuracy(logits, labels)
        self.log("test_acc", acc, on_step=False, on_epoch=True, logger=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        logits, loss = self._process_batch(batch, batch_idx)
        predictions = torch.argmax(logits, dim=-1)
        return predictions
