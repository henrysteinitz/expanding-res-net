import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes


class ResNetModule(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResNetModule, self).__init__()
        if kernel_size % 2 == 0:
            raise "Only odd kernel sizes are supported."
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        x2 = self.conv1(x)
        x2 = self.bn1(x2)
        x2 = F.relu(x2)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        # TODO: try batch norm
        x2 += x
        x2 = F.relu(x2)
        return x2


class ResNet(nn.Module):
    def __init__(self, layers=5, channels=20, kernel_size=3):
        super(ResNet, self).__init__()

        if kernel_size % 2 == 0:
            raise "Only odd kernel sizes are supported."
        padding = (kernel_size - 1) // 2

        # Initialize first few layers
        self.conv1 = nn.Conv2d(3, channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)

        # Initialize constant-size layers
        res_layers = []
        for _ in range(layers):
            res_layers.append(ResNetModule(channels))

        self.res = nn.Sequential(*res_layers)
        self.fc1 = nn.Linear(5120, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)

        # Apply ResNet modules
        x = self.res(x)

        # Apply final fully connected layers
        x = x.view(-1, 5120)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x











# 
import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class ResNetModule(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResNetModule, self).__init__()
        if kernel_size % 2 == 0:
            raise "Only odd kernel sizes are supported."
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        # We use batch norms to regularize.
        self.bn1 = nn.BatchNorm2d(channels, track_running_stats=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels, track_running_stats=False)

    def forward(self, x):
        x2 = self.conv1(x)
        x2 = self.bn1(x2)
        x2 = F.relu(x2)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 += x
        x2 = F.relu(x2)
        return x2


class ResNet(nn.Module):
    def __init__(self, layers1=6, layers2=8, layers3=6,  channels=43, kernel_size=3):
        super(ResNet, self).__init__()

        if kernel_size % 2 == 0:
            raise "Only odd kernel sizes are supported."
        padding = (kernel_size - 1) // 2

        # Initialize first layer
        self.conv1 = nn.Conv2d(3, channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels, track_running_stats=False)

        # Initialize constant-size residual layers
        res_layers1 = [ResNetModule(channels) for _ in range(layers1)]
        res_layers2 = [ResNetModule(channels) for _ in range(layers2)]


        self.res1 = nn.Sequential(*res_layers1)
        self.res2 = nn.Sequential(*res_layers2)
        self.linear_size = 256
        self.fc1 = nn.Linear(self.linear_size, 50)
        self.bn2 = nn.BatchNorm1d(50, track_running_stats=False)
        self.fc2 = nn.Linear(50, nclasses)

        

    def forward(self, x):
        # Apply a few convolutions with pooling to reduce 
        # the total number of parameters.
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        # Apply ResNet modules
        x = self.res1(x)
        x = F.max_pool2d(x, 2)
        x = self.res2(x)
        x = F.max_pool2d(x, 2)


        # Apply final fully connected layers
        x = x.view(-1, self.linear_size)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x



