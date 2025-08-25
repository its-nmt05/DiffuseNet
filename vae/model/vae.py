import torch
import torch.nn as nn
from vae.utils import get_norm, get_activation
from vae.model.blocks import upSampleBlock, downSampleBlock, attentionBlock


class Encoder(nn.Module):
    """VAE Encoder block"""

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.in_ch = config["in_ch"]
        self.down_channels = config["down_channels"]
        self.attn_channels = config["attn_channels"]
        self.num_heads = config["num_heads"]
        self.n_layers = config["n_layers"]
        self.norm_type = config["norm_type"]
        self.norm_groups = config["norm_groups"]
        self.activation = config["activation"]
        self.latent_dim = config["latent_dim"]

        self.conv_in = nn.Conv2d(self.in_ch, self.down_channels[0], kernel_size=3, stride=1, padding=1)
        self.encoder_blocks = nn.ModuleList()

        # downsampling blocks
        for i in range(len(self.down_channels) - 1):
            self.encoder_blocks.append(
                downSampleBlock(self.down_channels[i], self.down_channels[i + 1],
                                self.n_layers,
                                norm_type=self.norm_type,
                                norm_groups=self.norm_groups,
                                activation=self.activation)
            )

        # attention blocks
        for i in range(len(self.attn_channels) - 1):
            self.encoder_blocks.append(
                attentionBlock(self.attn_channels[i], self.attn_channels[i + 1],
                               self.n_layers,
                               num_heads=self.num_heads,
                               norm_type=self.norm_type,
                               norm_groups=self.norm_groups,
                               activation=self.activation)
            )

        self.norm = get_norm(self.norm_type, self.down_channels[-1], self.norm_groups)
        self.activation_layer = get_activation(self.activation)

        # conv layers to get mu and logvar
        self.conv_mu = nn.Conv2d(self.down_channels[-1], self.latent_dim, kernel_size=3, stride=1, padding=1)
        self.conv_logvar = nn.Conv2d(self.down_channels[-1], self.latent_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.encoder_blocks:
            x = block(x)
        x = self.norm(x)
        x = self.activation_layer(x)
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu


class Decoder(nn.Module):
    """VAE Decoder block"""

    def __init__(self, config):
        super(Decoder, self).__init__()
        self.in_ch = config["in_ch"]
        self.down_channels = config["down_channels"]
        self.attn_channels = config["attn_channels"]
        self.num_heads = config["num_heads"]
        self.n_layers = config["n_layers"]
        self.norm_type = config["norm_type"]
        self.norm_groups = config["norm_groups"]
        self.activation = config["activation"]
        self.latent_dim = config["latent_dim"]

        self.conv_in = nn.Conv2d(self.latent_dim, self.attn_channels[-1], kernel_size=3, stride=1, padding=1)
        self.decoder_blocks = nn.ModuleList()
        
        # attention blocks
        for i in reversed(range(len(self.attn_channels) - 1)):
            self.decoder_blocks.append(
                attentionBlock(self.attn_channels[i + 1], self.attn_channels[i],
                               self.n_layers,
                               num_heads=self.num_heads,
                               norm_type=self.norm_type,
                               norm_groups=self.norm_groups,
                               activation=self.activation)
            )

        # upsampling blocks
        for i in reversed(range(len(self.down_channels) - 1)):
            self.decoder_blocks.append(
                upSampleBlock(self.down_channels[i+ 1], self.down_channels[i],
                              self.n_layers,
                              norm_type=self.norm_type,
                              norm_groups=self.norm_groups,
                              activation=self.activation)
            )

        self.norm = get_norm(self.norm_type, self.down_channels[0], self.norm_groups)
        self.activation_layer = get_activation(self.activation)
        self.conv_out = nn.Conv2d(self.down_channels[0], self.in_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        x = self.conv_in(z)
        for block in self.decoder_blocks:
            x = block(x)
        x = self.norm(x)
        x = self.activation_layer(x)
        x = self.conv_out(x)
        x = nn.Sigmoid()(x)
        return x


class VAE(nn.Module):
    """VAE model"""

    def __init__(self, config):
        super(VAE, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        out = self.decoder(z)
        return out, mu, logvar
    
    def sample(self, num_samples=1):
        z = torch.randn(num_samples, self.encoder.latent_dim, 1, 1)
        return self.decoder(z)
    
    def encode(self, x):
        z, mu, logvar = self.encoder(x)
        return z, mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
