import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(1428, 174)
        
    def forward(self, x):
        x = torch.sigmoid(self.fc(x))
        return x
    