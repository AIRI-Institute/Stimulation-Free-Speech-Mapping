import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from constants import PROJECT_ROOT


def main(metrics_dir, plots_save_dir, low_freq, high_freq, step_freq, band_width, experiment, augmentation, models):
    assert low_freq > 0, low_freq
    assert high_freq > low_freq, (low_freq, high_freq)
    assert step_freq > 0, step_freq
    assert band_width > 0, band_width
    assert isinstance(low_freq, int), (type(low_freq), low_freq)
    assert isinstance(high_freq, int), (type(high_freq), high_freq)
    assert isinstance(step_freq, int), (type(step_freq), step_freq)
    assert isinstance(band_width, int), (type(band_width), band_width)

    half_width = band_width // 2
    freqs = list(range(low_freq, high_freq + 1, step_freq))
    n_freqs = len(freqs)

    aug_short_names = {
        'with_sound': 'with_sound',
        'without_sound': 'wo_sound'
    }
    lower_bounds = [None for _ in range(n_freqs)]
    upper_bounds = [None for _ in range(n_freqs)]

    aug_short = aug_short_names[augmentation]
    metrics_names = ['ROC', 'PR']

    per_model_maps = {}
    for idx1, c_freq1 in enumerate(freqs):
        for idx2, c_freq2 in enumerate(freqs):
            if c_freq1 > c_freq2:
                continue
            band_low = c_freq1 - half_width
            band_high = c_freq2 + half_width
            lower_bounds[idx1] = band_low
            upper_bounds[idx2] = band_high

            metrics = pd.read_csv(os.path.join(
                metrics_dir, f'freqs-{band_low}-{band_high}', experiment, augmentation,
                f'{experiment}_objects_{aug_short}_0.csv'
            ), index_col='Model')
            for model_name, metrics_row in metrics.iterrows():
                if model_name not in models:
                    continue
                if model_name not in per_model_maps:
                    per_model_maps[model_name] = {
                        'ROC': np.full((n_freqs, n_freqs), fill_value=np.nan),
                        'PR': np.full((n_freqs, n_freqs), fill_value=np.nan)
                    }
                for metric_name in metrics_names:
                    assert np.isnan(per_model_maps[model_name][metric_name][idx1, idx2]), (
                        idx1, idx2, c_freq1, c_freq2, per_model_maps[model_name][metric_name][idx1, idx2]
                    )
                    assert np.isnan(per_model_maps[model_name][metric_name][idx2, idx1]), (
                        idx1, idx2, c_freq1, c_freq2, per_model_maps[model_name][metric_name][idx2, idx1]
                    )
                    per_model_maps[model_name][metric_name][idx1, idx2] = metrics_row[metric_name]
    
    plt.rcParams.update({'font.size': 28})

    for model_name, model_metrics in per_model_maps.items():

        
        fig, ax = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

        for idx, metric_name in enumerate(metrics_names):
            metric_values = model_metrics[metric_name]

            # Draw the heatmap with the mask and correct aspect ratio
            sns.heatmap(metric_values, # mask=mask, 
                        cmap='viridis',
                        square=True, linewidths=.5, cbar_kws={'shrink': .8},
                        xticklabels=list(map(str, upper_bounds)), yticklabels=list(map(str, lower_bounds)), 
                        ax=ax[idx])

            ax[idx].set_title(f'{metric_name} AUC. Model: {model_name}')
            ax[idx].set_xlabel('Upper bound')
        
        ax[0].set_ylabel('Lower bound')

        plt.tight_layout()
        save_root = os.path.join(plots_save_dir, 'heatmaps', experiment, augmentation)
        Path(save_root).mkdir(exist_ok=True, parents=True)
        fig.savefig(os.path.join(save_root, f'{experiment}_{augmentation}_{model_name}.png'))
        plt.close(fig)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot frequency heatmaps')

    parser.add_argument(
        '--metrics-dir', type=str, help='A path to the folder (from the project root) with results',
        default='results/perc-test-trials-None', required=False
    )
    parser.add_argument(
        '--plots-save-dir', type=str, help='A path to the folder (from the project root) to save plots',
        default='visualization', required=False
    )
    parser.add_argument(
        '--low-freq', type=int, help='A lower bound of a potential central frequency', default=75, required=False
    )
    parser.add_argument(
        '--high-freq', type=int, help='A higher bound of a potential central frequency', default=175, required=False
    )
    parser.add_argument(
        '--step-freq', type=int, help='A step in frequencies between consecutive central frequencies', default=10,
        required=False
    )
    parser.add_argument(
        '--band-width', type=int, help='A width of a frequency band', default=50, required=False
    )
    parser.add_argument(
        '--experiment', type=str, help='Experiment to plot for',
        choices=('monopolar', 'bipolar'), default='monopolar', required=False
    )
    parser.add_argument(
        '--augmentation', type=str, help='Augmentation to plot for',
        choices=('with_sound', 'without_sound'), default='with_sound', required=False
    )
    parser.add_argument(
        '--models', type=str, nargs='+', help='Models to use',
        choices=('XGBoost', 'SVC', 'Logistic Regression', 'Random Forest', 'KNN', 'Decision Tree', 'MLP', 'AdaBoost', 'Gaussian Naive Bayes'),
        default=('SVC', ),
        required=False
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    main(
        metrics_dir=os.path.join(PROJECT_ROOT, args.metrics_dir),
        plots_save_dir=os.path.join(PROJECT_ROOT, args.plots_save_dir),
        low_freq=args.low_freq,
        high_freq=args.high_freq,
        step_freq=args.step_freq,
        band_width=args.band_width,
        experiment=args.experiment,
        augmentation=args.augmentation,
        models=args.models
    )
