import torch
import torch.nn as nn
import torch.distributions as td

from Project1.src.vae.flow import GaussianBase, MaskedCouplingLayer, Flow

class FlowPrior(nn.Module):
    def __init__(self, M, n_transformations, num_hidden=32):
        """
        Define a flow prior.

                Parameters:
        M: [int] 
            Dimension of the latent space.
        n_transformations: [int]
            Number of masked coupling layers.
        num_hidden: [int]
            Hidden dimension in scale and translation nets.
        """
        super(FlowPrior, self).__init__()
        self.M = M

        base = GaussianBase(M)
        transformations = []

        mask = torch.zeros(M)
        mask[M//2:] = 1 #Andre masking strategier?

        for _ in range(n_transformations):
            mask = (1-mask) #alternate

            scale_net = nn.Sequential(nn.Linear(M, num_hidden), nn.Tanh(), nn.Linear(num_hidden, M))
            translation_net = nn.Sequential(nn.Linear(M, num_hidden), nn.ReLU(), nn.Linear(num_hidden, M))

            nn.init.zeros_(scale_net[-1].weight)
            nn.init.zeros_(scale_net[-1].bias)
            nn.init.zeros_(translation_net[-1].weight)
            nn.init.zeros_(translation_net[-1].bias)

            transformations.append(MaskedCouplingLayer(scale_net=scale_net, translation_net=translation_net, mask=mask))

        self.flow = Flow(base=base, transformations=transformations)


    def forward(self):
        """
        Return the prior distribution, a pointer to this same class instance. 
        Allows for prior().log_prob(z) and prior.sample(N), for code generality.

        Returns:
        prior: [FlowPrior]
        """
        return self
    
    def log_prob(self, z):
        return self.flow.log_prob(z)
    
    def sample(self, sample_shape=torch.Size([])):
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)

        return self.flow.sample(sample_shape)
