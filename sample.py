import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import VariationalAutoencoder
from utils import display_images

plt.ion()

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_size = 12
N = 16 # number of samples to generate

model = VariationalAutoencoder()
model = model.to(device)
state_dict = torch.load('saves/vrnn_state_dict_16.pth')
model.load_state_dict(state_dict)
model.eval()

# sample the latent variable from a standard normal distribution
z = torch.randn((N, latent_size)).to(device)
# get the image samples
sample = model.decoder(z)

display_images(None, sample, N // 4, count=True)