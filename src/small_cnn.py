import torch.nn as nn


class SmallCNN(nn.Module):
    # A compact CNN for feature extraction and dimensionality reduction.
    def __init__(self, in_ch=3, out_ch=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.out = nn.Conv2d(128, out_ch, 1)

    def forward(self, x):
        x = self.net(x)
        x = self.out(x)
        return x.flatten(1)
