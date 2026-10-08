from .deepmedic import DeepMedic, SCALE_FACTORS, FEATURE_MAPS, FULLY_CONNECTED, DROPOUT
import torch.nn as nn
import torch
import torch.distributions as td
import torch.nn.functional as F


def pad(x, axis):
    shape = x.shape[:axis] + (1,) + x.shape[axis + 1 :]
    zeros = x.new_zeros(shape)
    return torch.concat((zeros, x), dim=axis)


class StochasticDeepMedic(DeepMedic):
    def __init__(
        self,
        input_channels,
        num_classes,
        scale_factors=SCALE_FACTORS,
        feature_maps=FEATURE_MAPS,
        fully_connected=FULLY_CONNECTED,
        dropout=DROPOUT,
        rank: int = 10,
        epsilon=1e-5,
        diagonal=False,
        use_zero_output=False,
        use_mask=True,
    ):
        super().__init__(
            input_channels,
            feature_maps[-1],
            scale_factors,
            feature_maps,
            fully_connected,
            dropout,
        )
        conv_fn = nn.Conv3d if self.dim == 3 else nn.Conv2d
        self.rank = rank
        self.num_classes = num_classes
        self.num_outputs = num_classes - 1 if use_zero_output else num_classes
        self.epsilon = epsilon
        self.diagonal = (
            diagonal  # whether to use only the diagonal (independent normals)
        )
        self.use_zero_output = use_zero_output
        self.use_mask = use_mask
        self.mean_l = conv_fn(
            feature_maps[-1], self.num_outputs, kernel_size=(1,) * self.dim
        )
        self.log_cov_diag_l = conv_fn(
            feature_maps[-1], self.num_outputs, kernel_size=(1,) * self.dim
        )
        self.cov_factor_l = conv_fn(
            feature_maps[-1], self.num_outputs * rank, kernel_size=(1,) * self.dim
        )

    def forward(self, image, **kwargs):
        logits = F.relu(super().forward(image, **kwargs)[0])
        batch_size = logits.shape[0]
        event_shape = (self.num_outputs,) + logits.shape[2:]

        mean = self.mean_l(logits)
        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = self.cov_factor_l(logits)
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_outputs, -1))
        cov_factor = cov_factor.flatten(2, 3)
        cov_factor = cov_factor.transpose(1, 2)

        # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI
        if self.use_mask:
            mask = kwargs["sampling_mask"]
            mask = (
                mask.unsqueeze(1)
                .expand((batch_size, self.num_outputs) + mask.shape[1:])
                .reshape(batch_size, -1)
            )
            cov_factor = cov_factor * mask.unsqueeze(-1)
            cov_diag = cov_diag * mask + self.epsilon
        else:
            cov_diag = cov_diag + self.epsilon

        # cov_factor = torch.zeros_like(cov_factor)
        # cov_diag = torch.full_like(cov_diag, 0.01) * mask + self.epsilon

        if self.diagonal:
            base_distribution = td.Independent(
                td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1
            )
        else:
            try:
                base_distribution = td.LowRankMultivariateNormal(
                    loc=mean, cov_factor=cov_factor, cov_diag=cov_diag
                )
            except:
                print(
                    "Covariance became not invertible using independent normals for this batch!"
                )
                base_distribution = td.Independent(
                    td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1
                )

        reshape_transform = td.transforms.ReshapeTransform(
            cov_diag.shape[1:], event_shape
        )
        distribution = td.TransformedDistribution(base_distribution, reshape_transform)

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        cov_diag_view = cov_diag.view(shape).detach()
        cov_factor_view = (
            cov_factor.transpose(2, 1)
            .view((batch_size, self.num_outputs * self.rank) + event_shape[1:])
            .detach()
        )

        if self.num_outputs != self.num_classes:

            # def pad(x):
            #     return torch.concat((torch.zeros_like(x[:, :1]), x), dim=1)

            logit_mean = pad(logit_mean, -4)
            cov_diag_view = pad(cov_diag_view, -4)
            padded_cov_factor_view = pad(
                cov_factor.reshape(
                    (batch_size, self.rank, self.num_outputs) + event_shape[1:]
                ),
                -4,
            )
            cov_factor_view = padded_cov_factor_view.view(
                (batch_size, self.num_classes * self.rank) + event_shape[1:]
            )

        output_dict = {
            "logit_mean": logit_mean.detach(),
            "cov_diag": cov_diag_view,
            "cov_factor": cov_factor_view,
            "distribution": distribution,
        }

        return logit_mean, output_dict
