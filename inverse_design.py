import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim=100, latent_dim=20, hidden_dim=50):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # latent_dim * 2 for mean and logvar
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

if __name__ == '__main__':
    # Quick test for the VAE module
    vae = VAE()
    sample_input = torch.randn((5, 100))
    reconstructed, mu, logvar = vae(sample_input)
    print("Input sample:", sample_input)
    print("Reconstructed:", reconstructed)
