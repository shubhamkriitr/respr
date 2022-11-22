from torch import nn
import torch
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, input, target):
        loss = torch.sqrt(self.mse(input, target) + self.eps)
        return loss