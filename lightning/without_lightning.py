# libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.dataset import random_split

# model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Linear(28 * 28, 128)
        self.layer2 = torch.nn.Linear(128, 10)
        self.log_smx = nn.LogSoftmax()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = self.log_smx(x)
        return x


# download data

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

# train dataloader
mnist_train = DataLoader(mnist_train, batch_size=64)

# val dataloader
mnist_val = DataLoader(mnist_val, batch_size=64)

# test dataloader
mnist_test = DataLoader(mnist_test, batch_size=64)

# optimizer + scheduler
net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

# train
for epoch in range(0, 100):
    net.train()
    for batch_idx, (data, target) in enumerate(mnist_train):
        # data, target = data.to(device),
        optimizer.zero_grad()
        output = net(data)
        # print("X->{0}".format(output.shape))
        # print("Y->{0}".format(target.shape))

        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print("Train Epoch :{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx*len(data), len(mnist_train.dataset),
                100 * batch_idx/len(mnist_train), loss.item()))

    # Validate
    net.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in mnist_test:
            output = net(data)
            test_loss +=F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct+=pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(mnist_test.dataset)

    print("\nTest set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(mnist_test.dataset), 100. * correct/len(mnist_test.dataset)
    ))








