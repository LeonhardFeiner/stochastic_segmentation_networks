import torch
from math import sqrt, pi
from torch.distributions import Distribution
from modified_bessel_function import modified_bessel_second_kind



class MultivariateLaplace(Distribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.dimension = len(loc)

    def log_prob(self, x):
        l2_norm = torch.norm(x - self.loc, p=2)
        z = sqrt(2) * l2_norm / self.scale
        log_prob = (
            -0.5 * self.dimension * torch.log(2 * self.scale ** 2)
            - 0.5 * self.dimension * torch.log(2 * pi)
            - z
            + torch.log(modified_bessel_second_kind(z, self.dimension / 2 - 1))
        )
        return log_prob

    def sample(self, sample_shape=torch.Size()):
        eps = torch.randn(*sample_shape, self.dimension, device=self.loc.device)
        return self.loc + self.scale * eps.sign() * (-torch.log(1 - 2 * eps.abs()))
