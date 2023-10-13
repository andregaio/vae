import torch
import torch.nn as nn
from layers.encoder import Encoder
from layers.decoder import Decoder


class VAE(nn.Module):
    def __init__(self, x_dim = 784, hidden_dim = 400, latent_dim = 200):
        super(VAE, self).__init__()

        self.encoder = Encoder(
            input_dim=x_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim = hidden_dim,
            output_dim = x_dim
        )

    def forward(self, x):
        mean, std = self.encoder(x)
        epsilon = torch.randn_like(std).to('cuda')
        x = mean + std * epsilon
        x = self.decoder(x)
        
        return x, mean, std