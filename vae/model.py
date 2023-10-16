import torch
import torch.nn as nn
from layers.encoder import Encoder
from layers.decoder import Decoder
from torchvision.utils import save_image


class VAE(nn.Module):
    def __init__(self, x_dim = 784, hidden_dim = 400, latent_dim = 200):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

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
    
    def generate_image(self, image_path = 'test/output_image.jpg'):
        device = next(self.parameters()).device
        noise = torch.randn(1, self.latent_dim).to(device)
        out = self.decoder(noise)
        save_image(out.view(1, 1, 28, 28), image_path)