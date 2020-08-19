import torch.nn as nn
class Model(nn.Module):
    """Some Information about Model"""
    def __init__(self):
        super().__init__()
        self.linear=nn.Sequential(
            nn.Linear(7,14),
            nn.ReLU(),
            nn.Linear(14,12),
            nn.ReLU(),
            nn.Linear(12,6),
            nn.ReLU(),
            nn.Linear(6,4),
            nn.ReLU(),
            nn.Linear(4,1)
        )
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x=self.linear(x)
        x=self.sigmoid(x)
        return x
