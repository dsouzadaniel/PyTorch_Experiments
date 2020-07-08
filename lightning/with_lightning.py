# libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.dataset import random_split

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics import Accuracy

class LitModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(28 * 28, 128)
        self.layer2 = torch.nn.Linear(128, 10)
        self.metric_accu = Accuracy(num_classes=10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        # Randomness
        # random_ixs = torch.randperm(n=len(x))
        # x = x[random_ixs]

        x = F.relu(x)
        x = self.layer2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        y_hat_argmax = y_hat.argmax(dim=1)
        pred = y_hat.argmax(dim=1)
        return {'loss': loss, 'true':y, 'pred':pred}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        all_true = torch.cat([x['true'] for x in outputs], dim=0).squeeze()
        all_pred = torch.cat([x['pred'] for x in outputs], dim=0).squeeze()
        #
        # print("X_true->{0}".format(all_true.shape))
        # print("Y_true->{0}".format(all_pred.shape))

        accuracy = self.metric_accu(all_pred, all_true)

        tensorboard_logs = {'loss/train': avg_loss, 'accu/train': accuracy, 'step': self.current_epoch}
        return {'loss/train': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1)
        return {'loss': loss, 'true':y, 'pred':pred}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        all_true = torch.cat([x['true'] for x in outputs], dim=0).squeeze()
        all_pred = torch.cat([x['pred'] for x in outputs], dim=0).squeeze()

        # print("X_val->{0}".format(all_true.shape))
        # print("Y_val->{0}".format(all_pred.shape))

        accuracy = self.metric_accu(all_pred, all_true)

        tensorboard_logs = {'loss/val': avg_loss,'accu/val': accuracy, 'step': self.current_epoch}
        return {'loss/val': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        loader = DataLoader(mnist_train, batch_size=64, num_workers=4, shuffle=True)
        return loader

    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
        loader = DataLoader(mnist_test, batch_size=64, num_workers=4, shuffle=False)
        return loader

from pytorch_lightning import Trainer

model = LitModel()

trainer = Trainer(max_epochs=10)

trainer.fit(model)
