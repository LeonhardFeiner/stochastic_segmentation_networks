"""Toy comparison: SSN Monte Carlo loss vs. analytic loss on smoothed labels.

Binary case, so the centered softmax is a sigmoid on a single logit and the
analytic loss is the Gaussian log-density of the smoothed target logits
log((1 - eps) / eps) * (2 * y - 1). The change-of-variables Jacobian depends only
on the target, so it is constant w.r.t. the parameters and is dropped.
"""
import argparse
import math

import torch
import torch.distributions as td

from toy_problem import (
    LowRankModel,
    get_on_off_binary_target,
    get_slide_bar_binary_target,
)

TARGETS = {"on_off": get_on_off_binary_target, "slide_bar": get_slide_bar_binary_target}


def mc_loss(dist, target, num_mc_samples):
    logit_sample = dist.rsample([num_mc_samples, target.shape[0]])
    target = target.expand((num_mc_samples,) + target.shape)
    log_prob = -torch.nn.functional.binary_cross_entropy_with_logits(
        logit_sample, target.double(), reduction="none"
    )
    loglikelihood = torch.logsumexp(log_prob.sum(-1), dim=0) - math.log(num_mc_samples)
    return -loglikelihood.mean()


def analytic_loss(dist, target, label_smoothing, cov_floor):
    target_logit = math.log((1 - label_smoothing) / label_smoothing) * (2 * target.double() - 1)
    gaussian = td.LowRankMultivariateNormal(
        dist.loc, dist.cov_factor, dist.cov_diag + cov_floor
    )
    return -gaussian.log_prob(target_logit).mean() / target.shape[-1]


def train(loss_name, target, args):
    torch.manual_seed(args.seed)
    model = LowRankModel(target.shape[-1], rank=args.rank)
    optimizer_mean = torch.optim.Adam([model.mean], lr=args.lr)
    optimizer_all = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step in range(args.steps):
        # as in toy_problem.py: optionally fit the mean alone before the covariance
        optimizer = optimizer_mean if step < args.mean_only_steps else optimizer_all
        optimizer.zero_grad()
        dist = model.get_dist()
        if loss_name == "mc":
            loss = mc_loss(dist, target, args.num_mc_samples)
        else:
            loss = analytic_loss(dist, target, args.label_smoothing, args.cov_floor)
        loss.backward()
        optimizer.step()
        if step % (args.steps // 5) == 0 or step == args.steps - 1:
            print(f"  {loss_name} step {step:5d} loss {loss.item():.4f}")
    return model.get_dist()


@torch.no_grad()
def evaluate(dist, target, num_samples=20000):
    masks = (dist.rsample([num_samples]) > 0).long()
    matches = (masks[:, None] == target[None]).all(-1)  # (samples, annotations)
    fractions = matches.double().mean(0)
    print(f"  P(sample == annotation k): {[round(f, 3) for f in fractions.tolist()]}")
    print(f"  P(sample matches none):    {1 - matches.any(-1).double().mean().item():.3f}")
    print(f"  pixel marginal P(y=1):     {[round(p, 2) for p in masks.double().mean(0).tolist()]}")
    print(f"  target marginal P(y=1):    {[round(p, 2) for p in target.double().mean(0).tolist()]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="on_off", choices=TARGETS)
    parser.add_argument("--dim", type=int, default=21)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--mean_only_steps", type=int, default=0)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--num_mc_samples", type=int, default=200)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--cov_floor", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    target = TARGETS[args.target](args.dim)
    print(f"target {args.target}: {target.shape[0]} annotations of {target.shape[1]} pixels")
    for loss_name in ["mc", "analytic"]:
        print(f"[{loss_name}]")
        evaluate(train(loss_name, target, args), target)
