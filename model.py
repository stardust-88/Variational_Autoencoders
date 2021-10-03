import torch
import torch.nn as nn

# Model class for Variational Autoencoder

# auxiliary-----------------------------------
class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    

class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :28, :28]
    

# model class------------------------------------
class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, stride=1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, stride=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1),
            nn.Flatten(),
        )
        
        self.z_mean = torch.nn.Linear(3136, latent_size)
        self.z_log_var = torch.nn.Linear(3136, latent_size)
        
        self.decoder = nn.Sequential(
            torch.nn.Linear(latent_size, 3136),
            Reshape(-1, 64, 7, 7),
            nn.ConvTranspose2d(64, 64, stride=1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3, padding=1),                
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, stride=2, kernel_size=3, padding=0),                
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, stride=1, kernel_size=3, padding=0), 
            Trim(),  # 1x29x29 -> 1x28x28
            nn.Sigmoid()
        )
        
    def reparameterize(self, z_mu, z_log_var):
    
        # sampling from a standard normal distribution
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.device)
    
        # creating a random variable z drawn from a normal distribution having parameters z_mu and z_log_var
        z = z_mu + eps*torch.exp(z_log_var/2.)
        return z
        
    def encode(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded, z_mean, z_log_var
    
    def forward(self, x):
        
        # encoding the given input x
        encoded, z_mean, z_log_var = self.encode(x)
        
        # decoding from the latent representaion 'encoded'
        decoded = self.decoder(encoded)
        
        return encoded, z_mean, z_log_var, decoded
        