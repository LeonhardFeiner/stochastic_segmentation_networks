import torch
import torch.nn.functional as F
import torch.nn as nn
import math


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
    ):
        super().__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, logits: torch.tensor, target: torch.tensor, **kwargs):
        return super().forward(logits, target)


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = torch.clamp(predictions, epsilon, 1.0 - epsilon)
    ce = -torch.mean(torch.log(predictions) * targets)
    return ce


def smooth_one_hot(target, axis, num_classes, label_smoothing, dtype):
    max_value = 1 - label_smoothing
    min_value = label_smoothing / (num_classes - 1)

    one_hot_target = torch.moveaxis(F.one_hot(target, num_classes), -1, axis)
    return torch.clamp(one_hot_target.type(dtype), min=min_value, max=max_value)


class StochasticSegmentationNetworkLossMCIntegral(nn.Module):
    def __init__(
        self,
        num_mc_samples: int = 1,
        label_smoothing=0,
        is_softmax=False,
        is_logsoftmax=False,
        softmax_axis=-2,
    ):
        super().__init__()
        self.num_mc_samples = num_mc_samples
        self.label_smoothing = label_smoothing
        self.is_softmax = is_softmax
        self.is_logsoftmax = is_logsoftmax
        self.softmax_axis = softmax_axis

    def forward(self, logits, target, distribution, **kwargs):
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        assert (
            num_classes >= 2
        )  # not implemented for binary case with implied background
        logit_sample = distribution.rsample((self.num_mc_samples,))
        target = target.unsqueeze(1)
        target = target.expand((self.num_mc_samples,) + target.shape)

        flat_size = self.num_mc_samples * batch_size
        logit_sample = logit_sample.view((flat_size, num_classes, -1))
        target = target.reshape((flat_size, -1))

        if self.is_softmax:
            # logit_sample = torch.log(logit_sample)

            smooth_labels = smooth_one_hot(
                target,
                self.softmax_axis,
                num_classes,
                self.label_smoothing,
                logit_sample.dtype,
            )
            log_prob_raw = torch.sum(
                torch.log(logit_sample) * smooth_labels, axis=self.softmax_axis
            )
            # if self.label_smoothing != 0:
            #     raise NotImplementedError()
            # log_prob_raw = -F.nll_loss(
            #     torch.log(logit_sample), target, reduction="none"
            # )
            loss_normalizer = log_prob_raw.shape[-1]

        elif self.is_logsoftmax:
            if self.label_smoothing:
                smooth_target = smooth_one_hot(
                    target,
                    self.softmax_axis,
                    num_classes,
                    self.label_smoothing,
                    logit_sample.dtype,
                )
                log_prob_raw = torch.sum(
                    logit_sample * smooth_target, axis=self.softmax_axis
                )
            else:
                log_prob_raw = -F.nll_loss(logit_sample, target, reduction="none")

            loss_normalizer = log_prob_raw.shape[-1]

        else:
            log_prob_raw = -F.cross_entropy(
                logit_sample,
                target,
                reduction="none",
                label_smoothing=self.label_smoothing,
            )
            loss_normalizer = 1

        log_prob = log_prob_raw.view((self.num_mc_samples, batch_size, -1))
        loglikelihood = torch.mean(
            torch.logsumexp(torch.sum(log_prob, dim=-1), dim=0)
            - math.log(self.num_mc_samples)
        )
        loss = -loglikelihood / loss_normalizer
        return loss


class StochasticSegmentationNetworkLossAnalytic(nn.Module):
    def __init__(self, label_smoothing, softmax_axis=-4, is_logsoftmax=False):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.softmax_axis = softmax_axis
        self.is_logsoftmax = is_logsoftmax

    def smooth_one_hot(self, target, num_classes, dtype):
        return smooth_one_hot(
            target, self.softmax_axis, num_classes, self.label_smoothing, dtype
        )

    def forward(self, logits, target, distribution, **kwargs):
        batch_size, num_classes, *remaining_shape = logits.shape
        assert (
            num_classes >= 2
        )  # not implemented for binary case with implied background

        loss_normalizer = torch.prod(logits.new_tensor(remaining_shape))

        smooth_target = self.smooth_one_hot(target, num_classes, logits.dtype)
        if self.is_logsoftmax:
            smooth_target = torch.log(smooth_target)
        return -torch.mean(distribution.log_prob(smooth_target)) / loss_normalizer
