import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import math
import torch.optim as optim

###################################################
class AdversarialNoiseGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, x):
        #generate nosie matching the size of x
        raise NotImplementedError()

class UniformNoiseGenerator(AdversarialNoiseGenerator):
    def __init__(self, min=0.0, max=1.0):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return (self.max - self.min) * torch.rand_like(x) + self.min

class NormalNoiseGenerator(AdversarialNoiseGenerator):
    def __init__(self, sigma=1.0, mu=0):
        super().__init__()
        self.sigma = sigma
        self.mu = mu

    def forward(self, x):
        return self.sigma * torch.randn_like(x) + self.mu
