import torch
import torch.nn as nn


def criterion(x, x_hat, mean, std, beta = 1):

    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    
    KLD = - 0.5 * torch.sum(1 + std - mean ** 2 - std.exp())
    
    return reproduction_loss + beta * KLD, reproduction_loss, beta * KLD