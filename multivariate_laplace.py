import torch
from torch.autograd import Function
from scipy.special import iv
import numpy as np
from torch.distributions import Laplace
from scipy.special import kn


class ModifiedBesselSecondKindFunction(Function):
    @staticmethod
    def forward(ctx, x, v):
        x_np = x.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        y_np = iv(v_np, x_np, is_scaled=False)
        y = torch.from_numpy(y_np).to(x.device)
        ctx.save_for_backward(x, v)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, v = ctx.saved_tensors
        x_np = x.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        grad_output_np = grad_output.detach().cpu().numpy()
        vjp = lambda v_: iv(v_, x_np, is_scaled=True)
        grad_x_np = grad_output_np * vjp(v_np)
        grad_v_np = grad_output_np * iv(v_np, x_np, derivative=1, is_scaled=True)
        grad_x = torch.from_numpy(grad_x_np).to(x.device)
        grad_v = torch.from_numpy(grad_v_np).to(v.device)
        return grad_x, grad_v


def modified_bessel_second_kind(x, v):
    return ModifiedBesselSecondKindFunction.apply(x, v)


class LowRankMultivariateLaplace(torch.distributions.Distribution):
    def __init__(self, loc, scale, rank):
        self.loc = torch.tensor(loc)
        self.scale = torch.tensor(scale)
        self.rank = rank

        # Define diagonal and low-rank covariance matrices
        diag = torch.diag(self.scale)
        self.U = torch.randn(len(self.loc), self.rank)
        self.S = torch.randn(self.rank)
        low_rank_cov = U @ torch.diag(S) @ U.t()

        # Compute the capacitance matrix
        self.capacitance_matrix = torch.inverse(diag + low_rank_cov)

        # Create a Laplace distribution with the scale parameter
        self.laplace = Laplace(0, self.scale)

    def sample(self, sample_shape=torch.Size()):
        # Sample from the Laplace distribution
        w = self.laplace.sample(sample_shape)
        # Sample from a standard normal distribution
        z = torch.randn(*(sample_shape + self.loc.shape))
        # Compute the final sample by adding the Laplace sample and the product of the capacitance matrix and the product of the transpose of the low-rank matrix and the standard normal sample
        return self.loc + w + self.capacitance_matrix @ (self.U.t() @ z)

    def log_prob(self, value):
        # Compute the log probability density function of the Laplace distribution
        laplace_log_prob = self.laplace.log_prob(value - self.loc)
        # Compute the Mahalanobis distance
        x = value - self.loc
        # Compute the dot product of the capacitance matrix and the Mahalanobis distance
        cap_dot_x = self.capacitance_matrix @ x
        # Compute the dot product of the Mahalanobis distance and the dot product of the capacitance matrix and the transpose of the low-rank matrix
        x_dot_U_cap = x @ (self.U @ self.capacitance_matrix)
        # Compute the Euclidean norm of the Mahalanobis distance
        x_norm = torch.norm(x, dim=-1)
        # Compute the modified Bessel function of the second kind
        n = len(self.loc)
        bessel = kn(n - 1, x_norm / self.scale) * ((x_norm / self.scale) ** (n / 2 - 1))
        # Compute the final log probability density function by summing the log probability density function of the Laplace distribution, the dot product of the Mahalanobis distance and the dot product of the capacitance matrix and the transpose of the low-rank matrix, and the logarithm of the modified Bessel function
        return (
            laplace_log_prob
            - (n / 2) * torch.log(2 * np.pi)
            + torch.log(bessel)
            - 0.5 * (x_dot_U_cap + cap_dot_x @ self.U.t() @ self.U @ cap_dot_x)
        )


import numpy as np
import scipy.stats
import torch

# Set the random seed
np.random.seed(0)
torch.manual_seed(0)

# Define the parameters of the LowRankMultivariateLaplace distribution
loc = [1, 2, 3]
scale = [1, 2, 3]
rank = 2

# Create the LowRankMultivariateLaplace distribution
dist = LowRankMultivariateLaplace(loc, scale, rank)

# Sample from the LowRankMultivariateLaplace distribution
samples = dist.sample(torch.Size([10000])).numpy()

# Compute the log probability density function of the LowRankMultivariateLaplace distribution
log_probs = dist.log_prob(torch.tensor(samples)).numpy()

# Create the MultivariateLaplace distribution from scipy.stats
multivariate_laplace = scipy.stats.multivariate_laplace(
    loc=loc, cov=np.diag(scale), k=rank
)

# Sample from the MultivariateLaplace distribution
scipy_samples = multivariate_laplace.rvs(size=10000)

# Compute the log probability density function of the MultivariateLaplace distribution
scipy_log_probs = multivariate_laplace.logpdf(scipy_samples)

# Compare the samples and log probability density function of the two distributions
print("Samples:")
print("LowRankMultivariateLaplace:", samples.mean(axis=0))
print("MultivariateLaplace:", scipy_samples.mean(axis=0))
print()
print("Log probability density function:")
print("LowRankMultivariateLaplace:", log_probs.mean())
print("MultivariateLaplace:", scipy_log_probs.mean())
