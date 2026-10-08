import torch
import torch.distributions as td
from torch.distributions.utils import _standard_normal
from torch.distributions.multivariate_normal import _batch_mv


class SymmetricLowRankMultivariateNormal(td.LowRankMultivariateNormal):
    def rsample(self, sample_shape=torch.Size()):
        first_shape_dim, *other_shape_dims = shape
        assert first_shape_dim % 2 == 0
        shape = self._extended_shape(
            torch.Size((first_shape_dim // 2, *other_shape_dims))
        )

        W_shape = shape[:-1] + self.cov_factor.shape[-1:]
        eps_W_raw = _standard_normal(
            W_shape, dtype=self.loc.dtype, device=self.loc.device
        )
        eps_D_raw = _standard_normal(
            shape, dtype=self.loc.dtype, device=self.loc.device
        )

        eps_W = torch.stack((eps_W_raw, -eps_W_raw), dim=0).reshape(-1, *W_shape[1:])
        eps_D = torch.stack((eps_D_raw, -eps_W_raw), dim=0).reshape(-1, *shape[1:])

        return (
            self.loc
            + _batch_mv(self._unbroadcasted_cov_factor, eps_W)
            + self._unbroadcasted_cov_diag.sqrt() * eps_D
        )


class UncentedLowRankMultivariateNormal(td.LowRankMultivariateNormal):
    diag_lambda_value = 1
    lr_lambda_value = 1

    def rsample(self, sample_shape=torch.Size()):
        target_sample_shape = torch.Size((self.cov_factor.shape[-1:], 3))
        assert sample_shape == target_sample_shape

        shape = self._extended_shape(sample_shape)
        W_shape = shape[:-1] + self.cov_factor.shape[-1:]

        eps_W_raw = torch.sqrt(self.lr_lambda_value + shape[:-1])
        eps_D_raw = torch.sqrt(self.lr_lambda_value + 1)
        eps_W = torch.broadcast_to(
            torch.stack([-eps_W_raw, 0, eps_W_raw], dim=0), (3, 3, *W_shape[1:])
        )
        eps_D = torch
        # eps_W = _standard_normal(W_shape, dtype=self.loc.dtype, device=self.loc.device)
        # eps_D = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return (
            self.loc
            + _batch_mv(self._unbroadcasted_cov_factor, eps_W)
            + self._unbroadcasted_cov_diag.sqrt() * eps_D
        )
