import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import set_all_seeds, display_images
from model import VariationalAutoencoder


# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# defining the loss function---------------------------------------------

def loss_fn(x_hat, x, mu, logvar):
    
    # calculating the reconstruction loss
    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')
    
    # calculating the KL divergence between the prior and the posterior distribution of the latent variable
    KLD = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD



# training--------------------------------------------------------------
def train():
    
    model.train()
    size = len(train_dataloader.dataset)
    train_loss = 0
    
    for batch_idx, (x, _) in enumerate(train_dataloader):
        x = x.to(device)
        
        # ----------forward--------------
        encoded, z_mean, z_log_var, decoded = model(x)
        loss = loss_fn(decoded, x, z_mean, z_log_var)
        train_loss += loss.item()
        
        # ----------backward--------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # ----------logging---------------
        if batch_idx % 100 == 0:
            current = batch_idx*len(x)
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]") 
        
    
    avg_train_loss = train_loss/size
    print(f"Training Data: \n Average loss: {avg_train_loss:>8f} \n")
    
    return avg_train_loss



# testing------------------------------------------------------------------
def test(epoch):
    
    model.eval()
    size = len(test_dataloader.dataset)
    test_loss = 0
    
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_dataloader):
            x = x.to(device)
        
            # ---------forward----------------
            encoded, z_mean, z_log_var, decoded = model(x)
            loss = loss_fn(decoded, x, z_mean, z_log_var)
            test_loss += loss.item()
            
    # --------logging--------------------------
    avg_test_loss = test_loss/size
    print(f"Test Data: \n Average loss: {avg_test_loss:>8f} \n")  
    display_images(x, decoded, 1, f'Epoch {epoch}')
    
    return avg_test_loss



# executing the model---------------------------------
def execute_model(epochs = 10):
    train_loss_per_epoch = []
    test_loss_per_epoch = []
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_epoch_loss = train()
        test_epoch_loss = test(epoch)
        
        train_loss_per_epoch.append(train_epoch_loss)
        test_loss_per_epoch.append(test_epoch_loss)
        
        #saving model
        if epoch % save_every == 1:
        fn = 'saves/vrnn_state_dict_'+str(epoch)+'.pth'
        torch.save(model.state_dict(), fn)
        print('Saved model to '+fn)
        
    print('execution completed')
    
    
    
# hyperparameters------------------------------------------
learning_rate = 0.001
batch_size = 64
num_epochs = 20 
latent_size = 10
#input_size = 28*28

# others----
save_every = 5



# load the the dataset and set up the dataloaders-----------
# loading the MNIST dataset
train_set = datasets.MNIST(root = 'data', train = True, download = False, transform = transforms.ToTensor())
test_set = datasets.MNIST(root = 'data', train = False, download = False, transform = transforms.ToTensor())

# Setting up the dataloaders
train_dataloader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
test_dataloader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle = True)
    

    
if __name__ == '__main__':
    
    random_seed = 123
    set_all_seeds(random_seed)
    
    # setting up the model and optimizer
    model = VariationalAutoencoder()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    execute_model(num_epochs)

