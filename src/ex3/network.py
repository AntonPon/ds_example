import torch
from torch import nn, flatten, Tensor
import torch.nn.functional as F




class BasicNet(nn.Module):
    def __init__(self, class_number=10, dtype=torch.float, *args, **kwargs) -> None:
        super(BasicNet, self).__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1, 32, 3,1, dtype=dtype)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, dtype=dtype)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)
        self.flattn = nn.Flatten(1)
        #self.dropout1 = nn.Dropout(0.25)
        #self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 64, dtype=dtype)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, class_number, dtype=dtype)


    def forward(self, x: Tensor)-> Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)
        x = flatten(x, 0 )
        x = self.fc1(x)
        x = self.relu3(x)
        #x = self.dropout2(x)
        x = self.fc2(x)
        return x