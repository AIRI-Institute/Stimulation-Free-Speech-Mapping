import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.constants import MAIN_RESULTS, PROJECT_ROOT, SUBJECTS, VISUALIZATIONS


def plot_single_pair(w, fig, w_ax, corr_ax, cbar_ax, feature_names, corr_matrix, x_times, x_ticks, n_steps_between_ticks):
    vmin, vmax = np.min(w), np.max(w)
    im_w = w_ax.imshow(w, cmap='viridis', aspect='auto')
    cbar = fig.colorbar(im_w, ax=w_ax, orientation='vertical')
    cbar_ticks = np.linspace(vmin, vmax, num=5)
    cbar.set_ticks(cbar_ticks)  # Set the number of ticks to be 5
    cbar.set_ticklabels([f'{v:.2f}' for v in cbar_ticks])  # Format tick labels
    w_ax.set_title(feature_names)
    
    # Set xticks every second step and append the final tick
    xticks = np.arange(0, len(x_times), step=n_steps_between_ticks)
    if xticks[-1] != (len(x_times) - 1):
        xticks = np.append(xticks, len(x_times) - 1)
    w_ax.set_xticks(xticks)
    w_ax.set_xticklabels(x_ticks, rotation=90)

    im_corr = corr_ax.imshow(corr_matrix, cmap='viridis')
    vmin, vmax = np.nanmin(corr_matrix), np.nanmax(corr_matrix)
    if cbar_ax is not None:
        cbar_ax = fig.add_axes(cbar_ax)
    cbar = fig.colorbar(im_corr, cax=cbar_ax, orientation='horizontal')
    cbar_ticks = np.linspace(vmin, vmax, num=5)
    cbar.set_ticks(cbar_ticks)  # Set the number of ticks to be 5
    cbar.set_ticklabels([f'{v:.2f}' for v in cbar_ticks])  # Format tick labels


def plot_weights(weights, corr_matrices, subjects, model, plots_save_dir, patterns, experiment, augmentation):
    feature_names = [
        'EEG relative to\nthe stimulus\n(0 - 3 s)',
        'EEG relative to\nthe beginning\nof speech (-1 - 1 s)',
        'Sound relative to\nthe stimulus\n(0 - 3 s)',
        'Sound relative to\nthe beginning\nof speech (-1 - 1 s)'
    ]
    n_steps_between_ticks = 5
    x_times = [
        np.round(np.linspace(0, 3, num=30), decimals=1),
        np.round(np.linspace(-1, 1, num=20), decimals=1),
        np.round(np.linspace(0, 3, num=30), decimals=1),
        np.round(np.linspace(-1, 1, num=20), decimals=1)
    ]
    x_ticks = [
        np.array([0, 0.5, 1, 1.5, 2, 2.5, 3]),
        np.array([-1, -0.5, 0, 0.5, 1]),
        np.array([0, 0.5, 1, 1.5, 2, 2.5, 3]),
        np.array([-1, -0.5, 0, 0.5, 1])
    ]

    assert len(weights) == len(corr_matrices), (len(weights), len(corr_matrices))

    n_columns = len(weights)

    all_weights = np.concatenate(weights, axis=1)
    norm_values = np.linalg.norm(all_weights, axis=1, keepdims=True)

    plt.rcParams.update({'font.size': 12})

    fig = plt.figure(figsize=(3 * n_columns, 8))
    subfigs = fig.subfigures(nrows=2, ncols=1)
    top_axes = subfigs[0].subplots(nrows=1, ncols=n_columns, sharey='row')
    bottom_axes = subfigs[1].subplots(nrows=1, ncols=n_columns)

    per_feature_figures = [plt.subplots(1, 2, figsize=(7, 4)) for _ in range(n_columns)]

    for idx in range(n_columns):
        w = weights[idx]
        w = w / norm_values
        
        plot_single_pair(
            w=w, fig=fig, w_ax=top_axes[idx], corr_ax=bottom_axes[idx], 
            cbar_ax=(0.125 + idx * 0.2022, 0.225, 0.1685, 0.01),
            feature_names=feature_names[idx], corr_matrix=corr_matrices[idx], 
            x_times=x_times[idx], x_ticks=x_ticks[idx], 
            n_steps_between_ticks=n_steps_between_ticks
        )
        bottom_axes[idx].axis('off')

        per_fig, per_ax = per_feature_figures[idx]

        plot_single_pair(
            w=w, fig=per_fig, w_ax=per_ax[0], corr_ax=per_ax[1], 
            cbar_ax=None,
            feature_names=f'A) {feature_names[idx]}', corr_matrix=corr_matrices[idx], 
            x_times=x_times[idx], x_ticks=x_ticks[idx], 
            n_steps_between_ticks=n_steps_between_ticks
        )
        if patterns:
            per_ax[1].set_title('B) Patterns correlation\ncoefficients')
        else:
            per_ax[1].set_title('B) Weights correlation\ncoefficients')
        per_ax[0].set_yticks(range(len(subjects)))
        per_ax[0].set_yticklabels([])
        per_ax[1].set_xticks(range(len(subjects)))
        per_ax[1].set_xticklabels([])
        per_ax[1].set_yticks(range(len(subjects)))
        per_ax[1].set_yticklabels([])
        per_ax[0].set_xlabel('t, sec.')
        per_ax[0].set_ylabel('Subject')
        per_ax[1].set_xlabel('Subject')
        per_ax[1].set_ylabel('Subject')

    top_axes[0].set_yticks(range(len(subjects)))
    top_axes[0].set_yticklabels(subjects)

    subfigs[1].subplots_adjust(top=1.35)

    if patterns:
        save_dir = os.path.join(plots_save_dir, 'feature_interpretation', 'patterns')
    else:
        save_dir = os.path.join(plots_save_dir, 'feature_interpretation', 'weights')
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'{experiment}_{augmentation}_{model}.png'), bbox_inches='tight')
    for idx, (fig, _) in enumerate(per_feature_figures):
        fig.savefig(os.path.join(save_dir, f'{experiment}_{augmentation}_{model}_per_feature_{idx}.png'), bbox_inches='tight')


def main(results_dir, low_freq, high_freq, experiment, augmentation, subjects, plots_save_dir, patterns, model):
    results_dir = os.path.join(
        results_dir, 'perc-test-trials-None', f'freqs-{low_freq}-{high_freq}', experiment, augmentation
    )
    if augmentation == 'with_sound':
        splits = [30, 20, 30, 20]
    elif augmentation == 'without_sound':
        splits = [30, 20]
    else:
        raise ValueError
    splits = [0] + splits
    splits = np.cumsum(splits).tolist()

    model_dir = os.path.join(results_dir, model)

    weights = []
    for _ in range(len(splits) - 1):
        weights.append([])

    for subject_name in subjects:
        sub_weights = np.load(os.path.join(model_dir, 'model_weights', f'model_sub_{subject_name}.npz'))['coef']
        sub_data = np.load(os.path.join(model_dir, 'data', f'data_sub_{subject_name}.npz'))['X_train']

        assert sub_weights.ndim == 2, sub_weights.shape
        assert sub_weights.shape[0] == 1, sub_weights.shape

        if patterns:
            cov_matrix = np.cov(sub_data.T)
            sub_weights = cov_matrix @ sub_weights.T  # weights into patterns
            sub_weights = sub_weights.T

        assert sub_weights.shape[0] == 1, sub_weights.shape
        sub_weights = sub_weights[0, :]

        for idx in range(len(splits) - 1):
            weights[idx].append(sub_weights[splits[idx]:splits[idx + 1]])

    for idx in range(len(splits) - 1):
        weights[idx] = np.stack(weights[idx], axis=0)

    all_corr_matrices = []
    for idx in range(len(splits) - 1):
        corrs_matrix = np.full((len(subjects), len(subjects)), fill_value=np.inf)
        for idx1 in range(len(subjects)):
            for idx2 in range(idx1, len(subjects)):
                w1 = weights[idx][idx1]
                w2 = weights[idx][idx2]
                corr = np.corrcoef(w1, w2)[0, 1]
                if corr < 0:
                    corr = np.corrcoef(w1, np.flip(w2))[0, 1]
                corrs_matrix[idx1, idx2] = corr
                corrs_matrix[idx2, idx1] = corr
        assert not np.isinf(corrs_matrix).any()
        all_corr_matrices.append(corrs_matrix)

    plot_weights(weights=weights, corr_matrices=all_corr_matrices, subjects=subjects, model=model,
                    plots_save_dir=plots_save_dir, patterns=patterns, experiment=experiment, augmentation=augmentation)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot topographies and weights')

    parser.add_argument(
        '--results-dir', type=str, help='A path to the folder (from the project root) to extract results',
        default=MAIN_RESULTS, required=False
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
        '--experiment', type=str, help='Experiment to plot for',
        choices=('monopolar', 'bipolar'), default='monopolar', required=False
    )
    parser.add_argument(
        '--augmentation', type=str, help='Augmentation to plot for',
        choices=('with_sound', 'without_sound'), default='with_sound', required=False
    )
    parser.add_argument(
        '--model', type=str, help='Model to use',
        choices=('xgboost', 'svc', 'logreg', 'random_forest', 'knn', 'decision_tree', 'mlp', 'adaboost', 'naive_bayes'),
        default='svc', required=False
    )
    parser.add_argument(
        '--patterns', action=argparse.BooleanOptionalAction,
        help='If True, turns weights into patterns before computations', default=True
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    main(
        results_dir=os.path.join(PROJECT_ROOT, args.results_dir), low_freq=args.low_freq,
        high_freq=args.high_freq, experiment=args.experiment, augmentation=args.augmentation, subjects=list(SUBJECTS),
        plots_save_dir=os.path.join(PROJECT_ROOT, args.plots_save_dir), patterns=args.patterns, model=args.model
    )
