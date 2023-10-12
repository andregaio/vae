import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to('cuda')
        z = mean + var*epsilon
        return z

    def forward(self, x):
        mean, std = self.Encoder(x)
        x = self.reparameterization(mean, torch.exp(0.5 * std))
        x = self.Decoder(x)
        
        return x, mean, std