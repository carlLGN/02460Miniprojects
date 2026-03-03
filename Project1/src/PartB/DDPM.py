# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm
from Project1.src.PartB.UnetClass import Unet 

class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """

        ### Implement Algorithm 1 here ###
        neg_elbo = 0
        batch_size = x.shape[0]
        device = x.device

        t = torch.randint(0, self.T, (batch_size,), device=device).long()

        #Sample epsilon 
        epsilon = torch.randn_like(x)

        alpha_bar_t = self.alpha_cumprod[t].view(batch_size, *([1] * (len(x.shape) - 1)))

        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        noisy_x = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * epsilon

        t_in = (t.float() / (self.T - 1)).unsqueeze(1)
        epsilon_theta = self.network(noisy_x, t_in)
        
        loss = F.mse_loss(epsilon_theta, epsilon, reduction='none')
        neg_elbo = loss.flatten(start_dim=1).sum(dim=-1)
        

        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        #Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)

        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T-1, -1, -1):
            ### Implement the remaining of Algorithm 2 here ###
            if t > 0:
                z = torch.randn_like(x_t)
            else:
                z = torch.zeros_like(x_t)
            alpha_bar_t = self.alpha_cumprod[t].view(1, *([1] * (len(shape) - 1)))
            
            # Create a time tensor of shape (batch_size, 1) to match x_t
            t_tensor = torch.full((x_t.shape[0], 1), t, device=self.alpha.device, dtype=torch.float)
    
            t_in = t_tensor / (self.T - 1)
            epsilon_theta = self.network(x_t, t_in)
            x_t = (x_t - (1 - self.alpha[t]) / torch.sqrt(1 - alpha_bar_t) * epsilon_theta) / torch.sqrt(self.alpha[t]) + torch.sqrt(self.beta[t]) * z

            pass

        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


def train(model, optimizer, train_loader, val_loader, epochs, device, patience=3, min_delta=1e-3):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for x in progress_bar:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)

            # Flattening for FcNetwork (remove if using a 2D Unet instead)
            x = x.view(x.size(0), -1)

            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():12.4f}")
            
        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                if isinstance(x, (list, tuple)):
                    x = x[0]
                x = x.to(device)
                
                # Apply the same flattening for validation
                x = x.view(x.size(0), -1)
                
                val_loss += model.loss(x).item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- Early Stopping Logic ---
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Optional: Save best model weights here
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered! Training stopped after {epoch+1} epochs.")
            break


class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden):
        """
        Initialize a fully connected network for the DDPM, where the forward function also take time as an argument.
        
        parameters:
        input_dim: [int]
            The dimension of the input data.
        num_hidden: [int]
            The number of hidden units in the network.
        """
        super(FcNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim+1, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, input_dim))

    def forward(self, x, t):
        """"
        Forward function for the network.
        
        parameters:
        x: [torch.Tensor]
            The input data of dimension `(batch_size, input_dim)`
        t: [torch.Tensor]
            The time steps to use for the forward pass of dimension `(batch_size, 1)`
        """
        x_t_cat = torch.cat([x, t], dim=1)
        return self.network(x_t_cat)


if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import argparse
    import os

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb', 'mnist'], help='dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    parser.add_argument('--model', type=str, default='ddpm_earlystop.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--n-samples', type=int, default=64)
    parser.add_argument('--patience', type=int, default=3, metavar='N', help='epochs to wait for improvement before stopping (default: %(default)s)')
    parser.add_argument('--min-delta', type=float, default=0.1, metavar='M', help='minimum required improvement (default: %(default)s)')
    args = parser.parse_args()

    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = torch.device(args.device)

    # Standard MNIST (non-binarized)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Unet 
    network = Unet()

    model = DDPM(network, T=args.T).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        train(
            model=model, 
            optimizer=optimizer, 
            train_loader=train_loader, 
            val_loader=test_loader, 
            epochs=args.epochs, 
            device=device,
            patience=args.patience,
            min_delta=args.min_delta
        )

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()

        with torch.no_grad():
            x = model.sample((args.n_samples, 28 * 28))        
            x = x.view(args.n_samples, 1, 28, 28)               
            x = (x + 1) / 2                                      
            x = x.clamp(0, 1)

        # 2x2 grid for 4 samples
        if args.n_samples == 4:
            nrow = 2
        else:
            nrow = int(args.n_samples ** 0.5)  

        save_image(x, args.samples, nrow=nrow)
        print("Saved samples to:", args.samples)