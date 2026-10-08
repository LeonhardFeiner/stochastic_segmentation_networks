"""Generalised energy distance and sample diversity as a function of sampling temperature.

Lighter than evaluate.py: a subset of cases and fewer samples, but the same image loading,
sampler and distance as the full evaluation. The temperature multiplies the standard deviation
of the Gaussian noise, so T=1 reproduces the model's own distribution.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from evaluator import Evaluator
from running_metrics.running_probability_distribution import calc_generalised_energy_distance
from running_metrics.samplers import LowRankMultivariateNormalTemperatureScaledRandomSampler

CLASS_NAMES = ['background', 'non-enhancing tumor', 'oedema', 'enhancing tumor']
EXTRA_MAPS = ['logit_mean', 'cov_diag', 'cov_factor']


def lesion_dice(samples, segmentation):
    samples = samples > 0
    segmentation = segmentation[None] > 0
    intersection = (samples & segmentation).reshape(len(samples), -1).sum(-1)
    total = samples.reshape(len(samples), -1).sum(-1) + segmentation.sum()
    return np.where(total > 0, 2 * intersection / np.maximum(total, 1), 1.0)


@torch.no_grad()
def sweep(csv_path, temperatures, num_cases, num_samples, device, block_size=2):
    evaluator = Evaluator(CLASS_NAMES, {}, target_name='seg', prediction_name='prediction',
                          mask_name='sampling_mask')
    rows = []
    for _, item in pd.read_csv(csv_path).head(num_cases).iterrows():
        _, segmentation, _, _, mask, extra_maps = evaluator.get_images(item, False, EXTRA_MAPS)
        for temperature in temperatures:
            sampler = LowRankMultivariateNormalTemperatureScaledRandomSampler(
                **extra_maps, device=device, mask=mask, temperature=temperature)
            # sample in blocks, as evaluate.py does, to bound GPU memory
            samples = np.concatenate([sampler(num_samples=block_size)[1].cpu().numpy().astype(np.uint8)
                                      for _ in range(num_samples // block_size)])
            del sampler
            torch.cuda.empty_cache()
            ged, _, diversity = calc_generalised_energy_distance(
                segmentation[None], samples, len(CLASS_NAMES), num_samples)
            rows.append({'id': item['id'], 'temperature': temperature, 'ged': ged,
                         'diversity': diversity,
                         'sample_lesion_dice': lesion_dice(samples, segmentation).mean()})
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-to-prediction-csv', required=True, type=str)
    parser.add_argument('--temperatures', default='1 0.5 0.25 0.1', type=str)
    parser.add_argument('--num-cases', default=10, type=int)
    parser.add_argument('--num-samples', default=20, type=int)
    parser.add_argument('--device', default=0, type=int)
    args = parser.parse_args()

    results = sweep(args.path_to_prediction_csv, [float(t) for t in args.temperatures.split()],
                    args.num_cases, args.num_samples, torch.device(args.device))
    output_path = os.path.join(os.path.dirname(args.path_to_prediction_csv), 'temperature_sweep.csv')
    results.to_csv(output_path, index=False)
    print(results.groupby('temperature')[['ged', 'diversity', 'sample_lesion_dice']].mean())
