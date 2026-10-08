from .deepmedic import DeepMedic, SCALE_FACTORS, FEATURE_MAPS, FULLY_CONNECTED, DROPOUT
import torch.nn as nn
import torch
import torch.distributions as td
import torch.nn.functional as F
from trainer.distributions import (
    CenteredLogSoftmaxTransform,
    CenteredSoftmaxTransform,
    PaddingTransform,
    LogSoftmaxTransform,
    SoftmaxTransform,
    pad,
    pad_epsilon,
)
from models.special_sampling_distributions import SymmetricLowRankMultivariateNormal


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
        use_softmax=False,
        use_log=False,
        use_symmetric_sampling=False,
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
        self.use_zero_output = use_zero_output
        self.use_mask = use_mask
        self.use_softmax = use_softmax
        self.use_log = use_log
        self.use_symmetric_sampling = use_symmetric_sampling
        self.num_outputs = self.num_classes - (1 if self.use_zero_output else 0)
        self.epsilon = epsilon
        self.diagonal = (
            diagonal  # whether to use only the diagonal (independent normals)
        )
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
        softmax_axis = 1 - logits.ndim

        mean = self.mean_l(logits)
        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = self.cov_factor_l(logits)
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_outputs, -1))
        asdf = cov_factor[..., 1000:1100]
        cov_factor = cov_factor.flatten(2, 3)
        cov_factor = cov_factor.transpose(1, 2)

        # cov_factor = torch.zeros_like(cov_factor)

        # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI
        if self.use_mask:
            mask = kwargs["sampling_mask"]
            mask = (
                mask.unsqueeze(1)
                .expand((batch_size, self.num_outputs) + mask.shape[1:])
                .reshape(batch_size, -1)
            )
            cov_factor = cov_factor * mask.unsqueeze(-1)
            cov_diag = cov_diag * mask + self.epsilon  # + (1 - mask)
        else:
            cov_diag = cov_diag + self.epsilon

        if self.diagonal:
            base_distribution = td.Independent(
                td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1
            )
        else:
            try:
                if self.use_symmetric_sampling:
                    base_distribution = SymmetricLowRankMultivariateNormal(
                        loc=mean, cov_factor=cov_factor, cov_diag=cov_diag
                    )
                else:
                    base_distribution = td.LowRankMultivariateNormal(
                        loc=mean, cov_factor=cov_factor, cov_diag=cov_diag
                    )
            except (RuntimeError, ValueError):
                print(
                    "Covariance became not invertible using independent normals for this batch!"
                )
                base_distribution = td.Independent(
                    td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1
                )

        transforms = [td.transforms.ReshapeTransform(cov_diag.shape[1:], event_shape)]
        if self.use_zero_output:
            if self.use_softmax:
                if self.use_log:
                    transforms.append(CenteredLogSoftmaxTransform(axis=softmax_axis))
                else:
                    transforms.append(CenteredSoftmaxTransform(axis=softmax_axis))
            else:
                transforms.append(PaddingTransform(axis=softmax_axis))
        else:
            if self.use_softmax:
                if self.use_log:
                    transforms.append(LogSoftmaxTransform(axis=softmax_axis))
                else:
                    transforms.append(SoftmaxTransform(axis=softmax_axis))

        distribution = td.TransformedDistribution(base_distribution, transforms)

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        cov_diag_view = cov_diag.view(shape).detach()
        cov_factor_view = (
            cov_factor.transpose(2, 1)
            .view((batch_size, self.num_outputs * self.rank) + event_shape[1:])
            .detach()
        )

        if self.use_zero_output:
            logit_mean = pad(logit_mean, softmax_axis)
            cov_diag_view = pad_epsilon(cov_diag_view, softmax_axis)
            padded_cov_factor_view = pad(
                cov_factor.reshape(
                    (batch_size, self.rank, self.num_outputs) + event_shape[1:]
                ),
                softmax_axis,
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
