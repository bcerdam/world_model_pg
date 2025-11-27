import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 32


class VAE(nn.Module):
    def __init__(self, z_dim=LATENT_DIM, img_channels=3):
        super(VAE, self).__init__()

        self.enc_conv1 = nn.Conv2d(img_channels, 32, 4, stride=2, padding=0)
        self.enc_conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)
        self.enc_conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=0)
        self.enc_conv4 = nn.Conv2d(128, 256, 4, stride=2, padding=0)

        self.fc_mu = nn.Linear(2 * 2 * 256, z_dim)
        self.fc_logvar = nn.Linear(2 * 2 * 256, z_dim)

        self.dec_fc1 = nn.Linear(z_dim, 1024)

        self.dec_conv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2, padding=0)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=0)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2, padding=0)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2, padding=0)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        x = F.relu(self.dec_fc1(z))
        x = x.view(-1, 1024, 1, 1)

        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        x = torch.sigmoid(self.dec_conv4(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar