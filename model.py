import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # input size: 3 x 128 x 128
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output size: 32 x 64 x 64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output size: 64 x 32 x 32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output size: 128 x 16 x 16
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 16 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, 2) # 2 outputs: Cat (0), Dog (1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.fc_layers(x)
        return x
