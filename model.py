import torch.nn as nn
from torchsummary import summary


class A2NN(nn.Module):
    def __init__(self, ):
        super(A2NN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(4*4*64, 9)

    def forward(self, inp):
        x = self.main(inp)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    nn = A2NN()
    summary(nn, (3, 32, 32))
