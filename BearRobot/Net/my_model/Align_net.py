import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignNet(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, embedding_dim)
        # self.fc3 = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x