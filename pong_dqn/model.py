import numpy as np 
import torch
# Create a model class torch
class PongModel(torch.nn.Module):
    def __init__(self):
        super(PongModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = torch.nn.Linear(9*9*32, 256)
        self.fc2 = torch.nn.Linear(256, 4)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # the input is 84 × 84 × 4 (height x width x 4 frames)
        x = self.conv1(x)
        x = self.relu(x)
        # now x is 20 × 20 × 16
        x = self.conv2(x)
        x = self.relu(x)
        # now x is 9 × 9 × 32
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # now x is 256
        x = self.relu(x)
        x = self.fc2(x)
        # now x is 4
        return x
if __name__ == "__main__":
    model = PongModel()
    sample = torch.zeros((1, 4, 84, 84))
    print(model(sample).shape)