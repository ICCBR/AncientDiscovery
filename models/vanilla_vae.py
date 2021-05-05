import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from base import BaseModel


def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


# ------------------------VanillaVAE----------------------------
class VanillaVAE(BaseModel):
    def __init__(self, in_channels=3, input_size=96, latent_dim=128, hidden_dims=5):
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = [2**i for i in range(5, hidden_dims+5)] if hidden_dims > 1 else [128]
        self.decode_hidden_dims = [2**i for i in range(hidden_dims+4, 4, -1)] if hidden_dims > 1 else [128]
        self.output_size = input_size
        self.image_size = input_size

        # Build Encoder
        modules = []
        channels = in_channels
        for h_dim in self.hidden_dims:
            layer = nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=h_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(h_dim),
                nn.ReLU()
            )
            modules.append(layer)
            channels = h_dim
            self.output_size = math.floor((self.output_size + 2 * 1 - 3) / 2 + 1)

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.hidden_dims[-1] * self.output_size ** 2, self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1] * self.output_size ** 2, self.latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dims[-1] * self.output_size ** 2)
        for i in range(len(self.decode_hidden_dims) - 1):
            layer = nn.Sequential(
                nn.ConvTranspose2d(self.decode_hidden_dims[i], self.decode_hidden_dims[i + 1], kernel_size=(3, 3),
                                   stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
                nn.BatchNorm2d(self.decode_hidden_dims[i + 1]),
                nn.ReLU()
            )
            modules.append(layer)
            output_size = (self.output_size - 1) * 2 + 3 - 2 * 1

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(self.decode_hidden_dims[-1], self.decode_hidden_dims[-1], kernel_size=(3, 3),
                               stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(self.decode_hidden_dims[-1]),
            nn.ReLU(),
            nn.ConvTranspose2d(self.decode_hidden_dims[-1], out_channels=in_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(in_channels),
            nn.Tanh()
        )

    def encode(self, image):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param image: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(image)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.decode_hidden_dims[0], self.output_size, self.output_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        result = F.interpolate(result, size=(self.image_size, self.image_size), mode='bilinear', align_corners=True)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, image):
        mu, log_var = self.encode(image)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), z, mu, log_var


class VanillaVAEMMD(VanillaVAE):
    def __init__(self, in_channels=3, input_size=96, latent_dim=128, hidden_dims=5):
        super(VanillaVAEMMD, self).__init__(in_channels, input_size, latent_dim, hidden_dims)
        self.mapping1 = nn.Linear(latent_dim, latent_dim * 2)
        self.mapping2 = nn.Linear(latent_dim * 2, latent_dim)
        self.activation = gelu
        self.dropout = nn.Dropout(0.2)

    def forward(self, image, img_from="source"):
        mu, log_var = self.encode(image)
        z = self.reparameterize(mu, log_var)
        if img_from == "target":
            z = self.mapping1(z)
        return self.decode(z), z, mu, log_var
