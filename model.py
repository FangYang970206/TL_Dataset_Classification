import torch.nn as nn
import numpy as np
from torchsummary import summary


class A2NN(nn.Module):
    def __init__(self, ):
        super(A2NN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.linear = nn.Linear(8*8*64, 10)

    def forward(self, inp):
        x = self.main(inp)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    inp = np.random.randn(3, 64, 64)
    nn = A2NN()
    summary(nn, (3, 64, 64))
