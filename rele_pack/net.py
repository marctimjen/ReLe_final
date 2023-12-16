import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.fn1 = nn.Linear(10, 50, dtype=torch.float64)
        self.fn2 = nn.Linear(50, 500, dtype=torch.float64)
        self.fn3 = nn.Linear(500, 90, dtype=torch.float64)
        self.fn4 = nn.Linear(90, 9, dtype=torch.float64)
        self.relu = nn.ReLU()

    def forward(self, x):

        out = self.fn1(x)
        out = self.relu(out)
        out = self.fn2(out)
        out = self.relu(out)
        out = self.fn3(out)
        out = self.relu(out)
        out = self.fn4(out)
        out = out.view(-1, 9)

        return out
