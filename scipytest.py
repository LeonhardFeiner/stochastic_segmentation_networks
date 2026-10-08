# %%
import scipy as scp

# %%
import numpy as np
import scipy.stats
import torch

# Set the random seed
np.random.seed(0)
# torch.manual_seed(0)

# Define the parameters of the LowRankMultivariateLaplace distribution
loc = [1, 2, 3]
scale = [1, 2, 3]
rank = 2

# Create the LowRankMultivariateLaplace distribution
# dist = LowRankMultivariateLaplace(loc, scale, rank)

# # Sample from the LowRankMultivariateLaplace distribution
# samples = dist.sample(torch.Size([10000])).numpy()

# # Compute the log probability density function of the LowRankMultivariateLaplace distribution
# log_probs = dist.log_prob(torch.tensor(samples)).numpy()

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
# print('LowRankMultivariateLaplace:', samples.mean(axis=0))
print("MultivariateLaplace:", scipy_samples.mean(axis=0))
print()
print("Log probability density function:")
# print('LowRankMultivariateLaplace:', log_probs.mean())
print("MultivariateLaplace:", scipy_log_probs.mean())

# %%
