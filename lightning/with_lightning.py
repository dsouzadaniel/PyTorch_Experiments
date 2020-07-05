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


class LitModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(28 * 28, 128)
        self.layer2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'train_loss': avg_loss, 'step': self.current_epoch}
        return {'train_loss': avg_loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        loader = DataLoader(mnist_train, batch_size=64, num_workers=4, shuffle=True)
        return loader


from pytorch_lightning import Trainer

model = LitModel()

trainer = Trainer()

trainer.fit(model)
