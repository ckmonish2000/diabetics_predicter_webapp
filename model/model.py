import torch.nn as nn
class Model(nn.Module):
    """Some Information about Model"""
    def __init__(self):
        super().__init__()
        self.linear=nn.Sequential(
            nn.Linear(7,4),
            nn.ReLU(),
            nn.Linear(4,2),
            nn.ReLU(),
            nn.Linear(2,1)            
        )
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x=self.linear(x)
        x=self.sigmoid(x)
        return x
