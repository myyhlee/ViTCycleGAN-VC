from pyclbr import Class
from pyparsing import Forward
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride = stride, padding=1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )


    def forward(self, x):
        return self.conv(x)


class Discriminator_CNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(  
                in_channels=1,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2)
        )

        self.model = nn.Sequential(
            Block(
                in_channels=64,
                out_channels=128,
                stride=2),
            Block(
                in_channels=128,
                out_channels=256,
                stride=2),
            Block(
                in_channels=256,
                out_channels=512,
                stride=1),
            nn.Conv2d(
                in_channels=512,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect")
        )

    
    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))


def test():
    x = torch.randn((5, 1, 80, 80))
    model = Discriminator_CNN(in_channels=1)
    preds = model(x)
    print(preds.shape)


if __name__=="__main__":
    test()



