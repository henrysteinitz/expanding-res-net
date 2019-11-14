import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes
class ResNetModule(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super(ResNetModule, self).__init__()
        if kernel_size % 2 == 0:
            raise "Only odd kernel sizes are supported."
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        # We use batch norms to regularize.
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        x2 = self.conv1(x)
        x2 = self.bn1(x2)
        x2 = F.relu(x2)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 += x
        x2 = F.relu(x2)
        return x2

# The channel expansion module also optionally includes an initial pooling operation.
class ExpansionModule(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True, kernel_size=5, padding=None):
        super(ExpansionModule, self).__init__()
        if kernel_size % 2 == 0:
            raise "Only odd kernel sizes are supported."
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        # We use batch norms to regularize.
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.pool = pool

    def forward(self, x):
        if self.pool:
            x = F.max_pool2d(x, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        return x

class Net10(nn.Module):
    def __init__(self, layer_depths=[1, 2, 2, 2]):
        super(Net10, self).__init__()
        channels = 32
        modules = [ExpansionModule(in_channels=3, out_channels=channels, pool=False, padding=0)]
        for depth in layer_depths:
            for i in range(depth):
                modules.append(ResNetModule(channels=channels))
            modules.append(ExpansionModule(in_channels=channels, out_channels=2*channels))
            channels *= 2
        self.res_layers = nn.Sequential(*modules)

        self.fc1 = nn.Linear(512, 1000)
        # self.fc2 = nn.Linear(1000, nclasses)


    def forward(self, x):
        x = self.res_layers(x)

        x = x.view(-1, 512)
        x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        return F.log_sofretmax(x)







