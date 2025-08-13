import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.constants import PROJECT_ROOT, TRIALS_RESULTS, VISUALIZATIONS


def add_metric_values(metric_values, metrics_path, print_perc_test,
                      models_to_use=('XGBoost', 'AdaBoost', 'SVC', 'LR')):
    models_stort_names = {
        'Random Forest': 'RF',
        'Decision Tree': 'DT',
        'Gaussian Naive Bayes': 'NB'
    }
    metrics = pd.read_csv(metrics_path, index_col='Model')

    for model_name, model_metrics in metrics.iterrows():
        if model_name not in models_to_use:
            continue
        if model_name in models_stort_names:
            model_name = models_stort_names[model_name]
        metric_values['Model'].append(model_name)
        metric_values['Percentage of trials'].append(print_perc_test)
        for metric_name in ['ROC', 'PR']:
            metric_values[f'{metric_name} AUC'].append(model_metrics[metric_name])


def main(bootstrap_results_dir, low_freq, high_freq, visualization_dir, perc_test_trials, n_bootstraps):
    results_dir = os.path.join(PROJECT_ROOT, bootstrap_results_dir)
    if n_bootstraps is None:
        visualization_dir = os.path.join(PROJECT_ROOT, visualization_dir, 'perc-test-trials')
        n_bootstraps = 1
        suffix = ''
    else:
        visualization_dir = os.path.join(PROJECT_ROOT, visualization_dir, 'perc-test-trials-bootstraps')
        suffix = '-bootstrap'
    Path(visualization_dir).mkdir(parents=True, exist_ok=True)

    perc_test_trials = sorted(perc_test_trials)
    x_ticks = []

    experiment = 'monopolar'
    augmentation = 'with_sound'
    metric_values = {'Model': [], 'ROC AUC': [], 'PR AUC': [], 'Percentage of trials': []}

    for perc_test in perc_test_trials:
        print_perc_test = 100 * perc_test
        x_ticks.append(int(print_perc_test))

        for b_idx in range(n_bootstraps):
            add_metric_values(
                metric_values=metric_values, metrics_path=os.path.join(
                    results_dir, f'perc-test-trials-{perc_test}' + suffix, f'freqs-{low_freq}-{high_freq}', experiment,
                    augmentation, f'{experiment}_objects_{augmentation}_{b_idx}.csv'
                ), print_perc_test=print_perc_test
            )

    plt.rcParams.update({'font.size': 26})
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    for idx, metric_name in enumerate(['ROC AUC', 'PR AUC']):

        sns.lineplot(
            data=metric_values, x='Percentage of trials', y=metric_name, errorbar='ci', hue='Model', ax=ax[idx],
            linewidth=3
        )
        ax[idx].set_title(f'{metric_name} vs. number of trials')
        ax[idx].legend(ncol=2)
        ax[idx].set_xticks(x_ticks)
        ax[idx].set_xticklabels(x_ticks)
    
    plt.tight_layout()
    fig.savefig(os.path.join(visualization_dir, f'{experiment}_{augmentation}.png'))
    plt.close(fig)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run experiments')

    parser.add_argument(
        '--bootstrap-metrics-dir', type=str, help='A path to the folder (from the project root) with bootstrap results',
        default=TRIALS_RESULTS, required=False
    )
    parser.add_argument(
        '--plots-save-dir', type=str, help='A path to the folder (from the project root) to save plots',
        default=VISUALIZATIONS, required=False
    )
    parser.add_argument(
        '--low-freq', type=int, help='A lower bound of a frequency band', default=100, required=False
    )
    parser.add_argument(
        '--high-freq', type=int, help='A higher bound of a frequency band', default=150, required=False
    )
    parser.add_argument(
        '--perc-test-trials', type=int, nargs='+',
        help='Percentages of test to append to the main graph with averaging over all trials',
        default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], required=False
    )
    parser.add_argument(
        '--n-bootstraps', type=int, help='Number of bootstraps to use. If None, no bootstrap is applied',
        required=False, default=100
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(
        bootstrap_results_dir=args.bootstrap_metrics_dir, 
        low_freq=args.low_freq, high_freq=args.high_freq, visualization_dir=args.plots_save_dir, 
        perc_test_trials=args.perc_test_trials, n_bootstraps=args.n_bootstraps
    )
