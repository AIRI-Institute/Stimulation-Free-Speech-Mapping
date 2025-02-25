import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, balanced_accuracy_score, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier

from constants import PROJECT_ROOT
from evaluate_models import parse_arguments, plot_results, run_trial


def compute_metrics(group, threshold):
    true_labels = group['label']
    pred_probs = group['prediction_proba']
    pred_labels = (pred_probs >= threshold).astype(int)

    TP = ((pred_labels == 1) & (true_labels == 1)).sum()
    FP = ((pred_labels == 1) & (true_labels == 0)).sum()
    TN = ((pred_labels == 0) & (true_labels == 0)).sum()
    FN = ((pred_labels == 0) & (true_labels == 1)).sum()

    sens = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    spec = TN / (TN + FP) if (TN + FP) > 0 else np.nan

    return pd.Series({'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN, 'Sens.': sens, 'Spec.': spec})


def create_results_table(results_df, threshold):
    # Apply the function to each subject
    metrics_per_subject = results_df.groupby(['subject_name']).apply(compute_metrics, threshold=threshold)

    metrics_per_subject.loc['mean'] = metrics_per_subject.mean()

    return metrics_per_subject


def run_classification(model_type, reg_type, reg_C, features, df_fit, subject_names):
    df_test_results = df_fit.copy()
    df_test_results['prediction_proba'] = 0.0

    for index, subject_name in enumerate(subject_names):
        df_train = df_fit.copy()
        df_train = df_train[df_train['subject_name'] != subject_name]

        df_test = df_fit.copy()
        df_test = df_test[df_test['subject_name'] == subject_name]

        X_train = features[df_train.index.values]
        X_test = features[df_test.index.values]

        y_train = df_train.label.values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if model_type == 'xgboost':
            clf = XGBClassifier()
        elif model_type == 'svc':
            clf = SVC(probability=True, kernel=reg_type, C=reg_C)
        elif model_type == 'logreg':
            if reg_type == 'l2':
                clf = LogisticRegression(C=reg_C)
            elif reg_type == 'l1':
                clf = LogisticRegression(penalty='l1', solver='liblinear', C=reg_C)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        clf.fit(X_train, y_train)

        y_hat_test = clf.predict_proba(X_test)[:, -1]

        df_test_results.loc[df_test_results['subject_name'] == subject_name, 'prediction_proba'] = y_hat_test
    return df_test_results


def evaluate_multiple_models(
    freq_band, fs, sr, downsample, hilbert_use,
    power_use, baseline_type, alignment_type, stimulus_time_seconds, prestimulus_time_start_seconds,
    prestimulus_time_finish_seconds, models, epochs_ieeg_trial, df_epochs_electrodes, subject_names,
    save_path, thresholds, epochs_sound_trial=None, print_message=''
):

    features, df_fit, df_results = run_trial(
        epochs_ieeg=epochs_ieeg_trial,
        df_epochs_electrodes=df_epochs_electrodes,
        freq_band=freq_band,
        fs=fs,
        sr=sr,
        downsample=downsample,
        hilbert_use=hilbert_use,
        power_use=power_use,
        baseline_type=baseline_type,
        alignment_type=alignment_type,
        stimulus_time_seconds=stimulus_time_seconds,
        prestimulus_time_start_seconds=prestimulus_time_start_seconds,
        prestimulus_time_finish_seconds=prestimulus_time_finish_seconds,
        bipolar=True,
        sound=False
    )

    if epochs_sound_trial is not None:
        features_sound, _, _ = run_trial(
            epochs_ieeg=epochs_sound_trial,
            df_epochs_electrodes=df_epochs_electrodes,
            freq_band=freq_band,
            fs=fs,
            sr=sr,
            downsample=downsample,
            hilbert_use=hilbert_use,
            power_use=power_use,
            baseline_type=baseline_type,
            alignment_type=alignment_type,
            stimulus_time_seconds=stimulus_time_seconds,
            prestimulus_time_start_seconds=prestimulus_time_start_seconds,
            prestimulus_time_finish_seconds=prestimulus_time_finish_seconds,
            bipolar=True,
            sound=True
        )
        features = np.concatenate([features, features_sound], axis=-1)

    print(print_message)

    results = {}
    for (model_type_, reg_type_, reg_C_) in tqdm(models, desc='Running classification for every model'):
        label = f'{model_type_}_{reg_type_}_{reg_C_}'
        df_test_results = run_classification(
            model_type=model_type_, reg_type=reg_type_, reg_C=reg_C_, features=features, df_fit=df_fit,
            subject_names=subject_names
        )
        results[label] = df_test_results

    rocs, prs = [], []

    for label, result in results.items():
        df_test_results = results[label].loc[df_fit.index]

        y_true = df_test_results['label'].values
        y_score = df_test_results['prediction_proba'].values

        fpr, tpr, thresholds_ftpr = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        rocs.append((label, roc_auc, fpr, tpr))

        precision, recall, thresholds_pr = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)

        prs.append((label, pr_auc, recall, precision))

        balanced_accuracy = balanced_accuracy_score(y_true, y_score > 0.5)
        print(label.ljust(20, '.'), balanced_accuracy)

    for threshold in thresholds:
        for label, df_test_results in results.items():
            metrics_per_subject = create_results_table(results_df=df_test_results, threshold=threshold)
            metrics_per_subject.to_csv(os.path.join(save_path, f'results_{label}_thres{threshold}.csv'))

    return results, rocs, prs


def run_bipolar_experiments(
    models, freq_band, fs, sr, downsample, hilbert_use,
    power_use, baseline_type, alignment_type, stimulus_time_seconds, prestimulus_time_start_seconds,
    prestimulus_time_finish_seconds, thresholds, dirpath_datasets, plots_save_dir, root_dir
):

    plots_save_dir = os.path.join(root_dir, plots_save_dir, 'bipolar')
    plots_save_dir_w_sound = os.path.join(plots_save_dir, 'with_sound')
    plots_save_dir_wo_sound = os.path.join(plots_save_dir, 'without_sound')
    os.makedirs(plots_save_dir_w_sound, exist_ok=True)
    os.makedirs(plots_save_dir_wo_sound, exist_ok=True)

    epochs_ieeg_trial = np.load(os.path.join(root_dir, dirpath_datasets, f'epochs_ieeg_bipolar{fs}_float32.npy'))
    epochs_sound = np.load(os.path.join(root_dir, dirpath_datasets, f'epochs_sound{fs}_float32.npy'))
    df_epochs_electrodes_bipolar = pd.read_csv(
        os.path.join(root_dir, dirpath_datasets, 'df_epochs_electrodes_bipolar.csv'), index_col=0
    )
    df_epochs_electrodes_bipolar['channel_name_1_2'] = df_epochs_electrodes_bipolar['channel_name_1'] + '-' + \
                                                       df_epochs_electrodes_bipolar['channel_name_2']

    subject_names = df_epochs_electrodes_bipolar['subject_name'].unique()

    epochs_sound_trial = np.zeros((df_epochs_electrodes_bipolar.shape[0], epochs_sound.shape[1]))
    for i in tqdm(range(epochs_sound.shape[0])):
        index = df_epochs_electrodes_bipolar[df_epochs_electrodes_bipolar.epoch_index_total == i].index.values
        epochs_sound_trial[index, :] = epochs_sound[i].reshape((1, -1))

    objects_with_sound = dict(
        save_path=plots_save_dir_w_sound,
        epochs_sound_trial=epochs_sound_trial,
        print_message='Bipolar, with sound',
        plots_name='bipolar_objects_with_sound.pdf'
    )

    objects_without_sound = dict(
        save_path=plots_save_dir_wo_sound,
        epochs_sound_trial=None,
        print_message='Bipolar, without sound',
        plots_name='bipolar_objects_wo_sound.pdf'
    )

    for params in (objects_without_sound, objects_with_sound):

        results, rocs, prs = evaluate_multiple_models(
            freq_band=freq_band,
            fs=fs,
            sr=sr,
            downsample=downsample,
            hilbert_use=hilbert_use,
            power_use=power_use,
            baseline_type=baseline_type,
            alignment_type=alignment_type,
            stimulus_time_seconds=stimulus_time_seconds,
            prestimulus_time_start_seconds=prestimulus_time_start_seconds,
            prestimulus_time_finish_seconds=prestimulus_time_finish_seconds,
            models=models, epochs_ieeg_trial=epochs_ieeg_trial, df_epochs_electrodes=df_epochs_electrodes_bipolar,
            subject_names=subject_names,
            save_path=params['save_path'],
            thresholds=thresholds,
            epochs_sound_trial=params['epochs_sound_trial'],
            print_message=params['print_message']
        )

        plot_results(
            rocs=rocs, prs=prs, save_path=os.path.join(params['save_path'], params['plots_name'])
        )

        print('\n')


if __name__ == '__main__':
    args = parse_arguments()

    models = [
        ('xgboost', None, None),
        ('svc', 'linear', 0.01),
        ('logreg', 'l2', 0.1),
        ('logreg', 'l1', 0.1),
    ]

    run_bipolar_experiments(
        models=models,
        freq_band=np.asarray([args.freq_band_low, args.freq_band_high]),
        fs=args.data_sr,
        sr=args.resample_sr,
        downsample=args.downsample,
        hilbert_use=args.hilbert_use,
        power_use=args.power_use,
        baseline_type=args.baseline_type,
        alignment_type=args.alignment_type,
        stimulus_time_seconds=args.stimulus_time_seconds,
        prestimulus_time_start_seconds=args.prestimulus_time_start_seconds,
        prestimulus_time_finish_seconds=args.prestimulus_time_finish_seconds,
        thresholds=args.thresholds,
        dirpath_datasets=args.datasets_dir_name,
        plots_save_dir=args.plots_save_dir,
        root_dir=PROJECT_ROOT
    )
