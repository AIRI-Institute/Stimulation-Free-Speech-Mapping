import argparse
import os

import einops
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier

from constants import ESM_NEGATIVE, ESM_POSITIVE, PROJECT_ROOT
from validation import get_average_roc, get_statistic, get_tables_subject


def create_relative_power(
        epochs, df, freq_band, fs, sr, downsample, hilbert_use, power_use, baseline_type, alignment_type,
        stimulus_time_seconds, prestimulus_time_start_seconds, prestimulus_time_finish_seconds, sound=False
):
    epoch_index = df.index.values
    epochs = np.copy(epochs[epoch_index])
    if not sound:
        b, a = sg.butter(4, freq_band, btype='bandpass', fs=fs)
        epochs = sg.filtfilt(b, a, epochs, axis=-1)

    if hilbert_use:
        epochs = np.abs(sg.hilbert(epochs, axis=-1))

    n_rows = epochs.shape[0]
    scale = np.arange(n_rows).reshape((-1, 1))

    prestimulus_time_start = round((stimulus_time_seconds + prestimulus_time_start_seconds) * fs)
    prestimulus_time_finish = round((stimulus_time_seconds + prestimulus_time_finish_seconds) * fs)

    if power_use:
        epochs = epochs ** 2
    else:
        epochs = np.abs(epochs)

    baseline = epochs[..., prestimulus_time_start:prestimulus_time_finish]

    baseline_mean = np.mean(baseline, axis=-1, keepdims=True)
    baseline_sd = np.std(baseline, axis=-1, keepdims=True)
    if baseline_type == 'log-averaged baseline':
        baseline_mean = np.mean(10 * np.log10(baseline + 1e-10), axis=-1, keepdims=True)

    def get_aligned(alignment_type):
        if alignment_type == 'stimulus':
            time_index = np.arange(round(fs * 3)).reshape((1, -1))
            epoch_starts = np.ones((n_rows, 1), dtype=np.int64) * round(stimulus_time_seconds * fs)
        elif alignment_type == 'voice':
            half_interval_time = round(fs * 1)
            time_index = np.arange(- half_interval_time, half_interval_time, 1).reshape((1, -1))
            epoch_starts = df['voice_start1000'].values.reshape((-1, 1))
        else:
            raise ValueError
        epoch_time_index = epoch_starts + time_index
        epochs_ = epochs[scale, epoch_time_index]
        if baseline_type == 'log-averaged baseline':
            epochs_ = 10 * np.log10(epochs_ + 1e-7)

        assert fs % sr == 0
        if downsample == 'average':
            ds_coef = round(fs / sr)
            epochs_ = einops.reduce(epochs_, 'b (t q) -> b t', 'mean', q=ds_coef)
        elif downsample == 'resample':
            epochs_ = librosa.resample(epochs_, orig_sr=fs, target_sr=sr)
        else:
            raise ValueError

        return epochs_

    if alignment_type == 'stimulus':
        epochs = get_aligned('stimulus')
    elif alignment_type == 'voice':
        epochs = get_aligned('voice')
    elif alignment_type == 'stimulus-voice':
        epochs1 = get_aligned('stimulus')
        epochs2 = get_aligned('voice')
        epochs = np.concatenate([epochs1, epochs2], axis=-1)
    else:
        raise ValueError

    if baseline_type == 'absolute baseline':
        epochs = epochs - baseline_mean
    elif baseline_type == 'z-score baseline':
        epochs = (epochs - baseline_mean) / baseline_sd
    elif baseline_type == 'relative baseline':
        epochs = epochs / baseline_mean
    elif baseline_type == 'log-transform baseline':
        epochs = 10 * np.log10(epochs / baseline_mean)
    elif baseline_type == 'absolute-relative baseline':
        epochs = (epochs - baseline_mean) / baseline_mean
    elif baseline_type == 'normed baseline':
        epochs = (epochs - baseline_mean) / (epochs + baseline_mean)
    elif baseline_type == 'log-averaged baseline':
        epochs = epochs - baseline_mean
    else:
        raise ValueError

    return epochs


def create_df_results(df, bipolar):
    if bipolar:
        channel_name_column = 'channel_name_1_2'
    else:
        channel_name_column = 'channel_name'
    groupby = ['subject_id', channel_name_column]
    df_ = df.groupby(groupby).size().reset_index(name='size')
    df_results = df.reset_index(drop=True).groupby(groupby).apply(lambda x: np.asarray(x.index.tolist()),
                                                                  include_groups=False).reset_index(name='indices')
    df_results['size'] = df_['size']
    return df_results


def logpower_results(epochs, df, bipolar):
    df_results = create_df_results(df, bipolar=bipolar)
    results = np.zeros((df_results.shape[0], epochs.shape[-1]))
    for i, row in df_results.iterrows():
        indices = row['indices']
        epochs_channel_mean = epochs[indices].mean(axis=0)
        results[i] = epochs_channel_mean
    return results, df_results


def run_trial(
        epochs_ieeg, df_epochs_electrodes, freq_band, fs, sr, downsample, hilbert_use,
        power_use, baseline_type, alignment_type, stimulus_time_seconds, prestimulus_time_start_seconds,
        prestimulus_time_finish_seconds, bipolar, sound=False
):
    df_ieeg = df_epochs_electrodes.copy()
    df_ieeg = df_ieeg[df_ieeg['voice_valid'] == 1]
    df_ieeg = df_ieeg[df_ieeg['stimulus_type'] == 'objects']

    epochs = create_relative_power(
        epochs=epochs_ieeg, df=df_ieeg,
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
        sound=sound
    )
    features, _ = logpower_results(epochs, df_ieeg, bipolar=bipolar)

    df_results = df_ieeg.copy()
    if bipolar:
        groupby = ['subject_name', 'subject_id', 'channel_name_1_2', 'esm_type']
    else:
        groupby = ['subject_name', 'subject_id', 'channel_name', 'esm_type', 'esm_result_info', 'esm_result_concensus']
    df_results = df_results.reset_index(drop=True).groupby(groupby).apply(lambda x: np.asarray(x.index.tolist()),
                                                                          include_groups=False).reset_index(
        name='indices')

    df_fit = df_results.copy()
    if bipolar:
        df_fit = df_fit[df_fit['esm_type'].isin(set(ESM_NEGATIVE) | set(ESM_POSITIVE))]
        df_fit['label'] = df_fit['esm_type'].isin(ESM_POSITIVE).astype(int)
        df_fit = df_fit.drop(columns=['indices', 'esm_type'])
    else:
        esm_type = list(ESM_POSITIVE)
        df_fit = df_fit[df_fit[f'esm_result_info'] == 1]
        df_fit = df_fit[df_fit['esm_type'].isin([0] + esm_type)]
        df_fit['label'] = 0
        df_fit.loc[df_fit['esm_type'].isin(esm_type), 'label'] = 1
        df_fit = df_fit.drop(
            columns=['esm_type', 'esm_result_info', 'esm_result_concensus', 'indices', 'esm_result_info']
        )

    return features, df_fit, df_results


def get_average_pr(tables, thresholds, subject_names):
    pr = np.zeros(len(thresholds))
    rec = np.zeros(len(thresholds))
    for i, thr in enumerate(thresholds):
        table_total = {
            'p+|a+': 0,
            'p-|a+': 0,
            'p+|a-': 0,
            'p-|a-': 0,
        }

        for subject_name in subject_names:
            table = tables[subject_name][i]
            for k, v in table.items():
                table_total[k] += v

        sensitivity, specificity, f1score, precision, falsepr = get_statistic(table_total)
        pr[i] = precision
        rec[i] = sensitivity
    return rec, pr


def get_average_ss(tables, thresholds, subject_names):
    senc = np.zeros(len(thresholds))
    spec = np.zeros(len(thresholds))
    for i, thr in enumerate(thresholds):
        table_total = {
            'p+|a+': 0,
            'p-|a+': 0,
            'p+|a-': 0,
            'p-|a-': 0,
        }

        for subject_name in subject_names:
            table = tables[subject_name][i]
            for k, v in table.items():
                table_total[k] += v

        sensitivity, specificity, f1score, precision, falsepr = get_statistic(table_total)
        senc[i] = sensitivity
        spec[i] = specificity
    return senc, spec


def get_roc_auc(fpr, tpr):
    fpr_auc = np.concatenate([fpr[::-1], [1]])
    tpr_auc = tpr[::-1]

    tpr_auc[0] = 0
    for i in range(1, tpr_auc.shape[0]):
        if np.isnan(tpr_auc[i]):
            tpr_auc[i] = tpr_auc[i - 1]

    fpr_auc_diff = np.diff(fpr_auc)
    roc_auc = np.sum(fpr_auc_diff * tpr_auc)
    return roc_auc


def get_pc_auc(rec, pr):
    rec_auc = rec[::-1]
    pr_auc = pr[::-1]

    pr_auc[0] = 1
    for i in range(1, pr_auc.shape[0]):
        if np.isnan(pr_auc[i]):
            pr_auc[i] = pr_auc[i - 1]

    pr_auc_ave = 1 / 2 * (pr_auc[1:] + pr_auc[:-1])
    rec_auc_diff = np.diff(rec_auc)

    pc_auc = np.sum(rec_auc_diff * pr_auc_ave)
    return pc_auc


def get_ss_auc(senc, spec):
    senc_auc = np.concatenate([senc[::-1], [1]])
    spec_auc = spec[::-1]

    spec_auc[0] = 1
    for i in range(1, spec_auc.shape[0]):
        if np.isnan(spec_auc[i]):
            spec_auc[i] = spec_auc[i - 1]
    senc_auc_diff = np.diff(senc_auc)
    ss_auc = np.sum(senc_auc_diff * spec_auc)
    return ss_auc


def create_results_table(per_subject_tables, thresholds):

    results_tables = {}

    for thres_idx, threshold in enumerate(thresholds):

        results_table = {
            'TP': [], 'FN': [], 'FP': [], 'TN': [], 'Sens.': [], 'Spec.': []
        }
        subject_names = []

        for subject, subject_thres_table in per_subject_tables.items():
            subject_table = subject_thres_table[thres_idx]
            subject_names.append(subject)

            results_table['TP'].append(subject_table['p+|a+'])
            results_table['FN'].append(subject_table['p-|a+'])
            results_table['FP'].append(subject_table['p+|a-'])
            results_table['TN'].append(subject_table['p-|a-'])

            if (subject_table['p+|a+'] + subject_table['p-|a+']) != 0:
                sensitivity = subject_table['p+|a+'] / (subject_table['p+|a+'] + subject_table['p-|a+'])
            else:
                sensitivity = np.nan
            if (subject_table['p-|a-'] + subject_table['p+|a-']) != 0:
                specificity = subject_table['p-|a-'] / (subject_table['p-|a-'] + subject_table['p+|a-'])
            else:
                specificity = np.nan
            results_table['Sens.'].append(sensitivity)
            results_table['Spec.'].append(specificity)

        results_table = pd.DataFrame(results_table, index=subject_names)
        results_table.loc['mean'] = results_table.mean()
        results_tables[threshold] = results_table

    return results_tables


def run_classification(model_type, reg_type, reg_C, features, df_fit, df_results, subject_names):
    df_test_results = df_results.copy()

    for index, subject_name in enumerate(subject_names):
        df_train = df_fit.copy()
        df_train = df_train[df_train['subject_name'] != subject_name]

        df_test = df_results.copy()
        df_test = df_test[df_test['subject_name'] == subject_name]
        df_test = df_test.drop(
            columns=['esm_type', 'esm_result_info', 'esm_result_concensus', 'indices', 'esm_result_info'])

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
        prestimulus_time_finish_seconds, models, epochs_ieeg_trial, df_epochs_electrodes, subject_names, dfs_active,
        save_path, thresholds_cm, epochs_sound_trial=None, print_message=''
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
        bipolar=False,
        sound=False
    )

    if epochs_sound_trial is not None:
        features_sound, df_fit_, df_results_ = run_trial(
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
            bipolar=False,
            sound=False
        )
        features = np.concatenate([features, features_sound], axis=-1)

    print(print_message)

    results = {}

    rocs, prs = [], []

    for (model_type_, reg_type_, reg_C_) in models:
        df_test_results = run_classification(
            model_type=model_type_, reg_type=reg_type_, reg_C=reg_C_, features=features, df_fit=df_fit,
            df_results=df_results, subject_names=subject_names
        )

        esm_negative_ = set(str(esm_value) for esm_value in ESM_NEGATIVE)
        esm_positive_ = set(str(esm_value) for esm_value in ESM_POSITIVE)
        thresholds = np.concatenate([[0], np.sort(df_test_results['prediction_proba'].values), [1]])
        tables = get_tables_subject(df_test_results, thresholds, subject_names, dfs_active, esm_negative_,
                                    esm_positive_)

        fpr, tpr = get_average_roc(tables, thresholds, subject_names)
        roc_auc = get_roc_auc(fpr, tpr)
        rec, pr = get_average_pr(tables, thresholds, subject_names)
        pc_auc = get_pc_auc(rec, pr)
        senc, spec = get_average_ss(tables, thresholds, subject_names)
        ss_auc = get_ss_auc(senc, spec)

        label = f'{model_type_}_{reg_type_}_{reg_C_}'
        print(f'{label}, roc_auc {roc_auc}, pc_auc {pc_auc}, ss_auc {ss_auc}')
        rocs.append((label, roc_auc, fpr, tpr))
        prs.append((label, pc_auc, rec, pr))

        results[label] = df_test_results

    for label, df_test_results in results.items():
        esm_negative_ = set(str(esm_value) for esm_value in ESM_NEGATIVE)
        esm_positive_ = set(str(esm_value) for esm_value in ESM_POSITIVE)
        tables = get_tables_subject(df_test_results, thresholds_cm, subject_names, dfs_active, esm_negative_,
                                    esm_positive_)
        results_tables = create_results_table(per_subject_tables=tables, thresholds=thresholds_cm)
        for threshold, metrics_per_subject in results_tables.items():
            metrics_per_subject.to_csv(os.path.join(save_path, f'results_{label}_thres{threshold}.csv'))

    return results, rocs, prs


def plot_results(rocs, prs, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    xlabels = ['False Positive Rate', 'Recall']
    ylabels = ['True Positive Rate', 'Precision']
    titles = ['ROC Curve', 'PR Curve']
    locs = ['lower right', 'lower left']
    labels = {
        'xgboost_None_None': 'XGBoost',
        'svc_linear_0.01': 'Linear SVC',
        'logreg_l2_0.1': 'Logistic Regression + L2 penalty',
        'logreg_l1_0.1': 'Logistic Regression + L1 penalty',
    }

    print('ROC')
    for label, roc_auc, fpr, tpr, in rocs:
        ax[0].plot(fpr, tpr, label=f'{labels[label]}')
        print(labels[label].ljust(40, '.') + str(round(roc_auc, 4)))

    print('PR')
    for label, pc_auc, recall, precision in prs:
        ax[1].plot(recall, precision, label=f'{labels[label]}')
        print(labels[label].ljust(40, '.') + str(round(pc_auc, 4)))

    for i in range(2):
        ax[i].set_xlim(0, 1)
        ax[i].set_ylim(0, 1)
        ax[i].legend(loc=locs[i])
        ax[i].set_xlabel(xlabels[i])
        ax[i].set_ylabel(ylabels[i])
        ax[i].set_title(titles[i])

    fig.tight_layout()
    fig.savefig(save_path)

    return fig, ax


def run_monopolar_experiments(
    models, freq_band, fs, sr, downsample, hilbert_use,
    power_use, baseline_type, alignment_type, stimulus_time_seconds, prestimulus_time_start_seconds,
    prestimulus_time_finish_seconds, thresholds, dirpath_datasets, plots_save_dir, root_dir, raw_directory
):

    plots_save_dir = os.path.join(root_dir, plots_save_dir, 'monopolar')
    plots_save_dir_w_sound = os.path.join(plots_save_dir, 'with_sound')
    plots_save_dir_wo_sound = os.path.join(plots_save_dir, 'without_sound')
    os.makedirs(plots_save_dir_w_sound, exist_ok=True)
    os.makedirs(plots_save_dir_wo_sound, exist_ok=True)

    epochs_ieeg_trial = np.load(os.path.join(root_dir, dirpath_datasets, f'epochs_ieeg{fs}_float32.npy'))
    epochs_sound = np.load(os.path.join(root_dir, dirpath_datasets, f'epochs_sound{fs}_float32.npy'))
    df_epochs_electrodes = pd.read_csv(
        os.path.join(root_dir, dirpath_datasets, 'df_epochs_electrodes.csv'), index_col=0
    )

    subject_names = df_epochs_electrodes['subject_name'].unique()
    dfs_active = {}
    for subject_name in tqdm(subject_names, desc='Loading subjects data'):
        subject_info = os.path.join(root_dir, raw_directory, 'sheets', 'subjects_sheets', f'{subject_name}.xlsx')
        dfs_active[subject_name] = pd.read_excel(subject_info, sheet_name='Active Stimulation')[
            ['Channel 1', 'Channel 2', 'Result']]

    df_epochs_electrodes = pd.read_csv(
        os.path.join(root_dir, dirpath_datasets, 'df_epochs_electrodes.csv'), index_col=0
    )

    df_epochs_electrodes.loc[pd.isna(df_epochs_electrodes['voice_start1000'])] = -1
    df_epochs_electrodes['voice_start1000'] = df_epochs_electrodes['voice_start1000'].astype(int)

    epochs_sound_trial = np.zeros((df_epochs_electrodes.shape[0], epochs_sound.shape[1]))
    for i in tqdm(range(epochs_sound.shape[0]), desc='Collecting epochs for sounds'):
        index = df_epochs_electrodes[df_epochs_electrodes.epoch_index_total == i].index.values
        epochs_sound_trial[index, :] = epochs_sound[i].reshape((1, -1))

    objects_with_sound = dict(
        save_path=plots_save_dir_w_sound,
        epochs_sound_trial=epochs_sound_trial,
        print_message='Monopolar, with sound',
        plots_name='monopolar_objects_with_sound.pdf'
    )

    objects_without_sound = dict(
        save_path=plots_save_dir_wo_sound,
        epochs_sound_trial=None,
        print_message='Monopolar, without sound',
        plots_name='monopolar_objects_wo_sound.pdf'
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
            models=models, epochs_ieeg_trial=epochs_ieeg_trial, df_epochs_electrodes=df_epochs_electrodes,
            subject_names=subject_names, dfs_active=dfs_active,
            save_path=params['save_path'],
            thresholds_cm=thresholds,
            epochs_sound_trial=params['epochs_sound_trial'],
            print_message=params['print_message']
        )

        plot_results(
            rocs=rocs, prs=prs, save_path=os.path.join(params['save_path'], params['plots_name'])
        )

        print('\n')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run experiments')

    parser.add_argument(
        '--datasets-dir-name', type=str, help='A path to the folder (from the project root) to load dataset',
        default='datasets_preprocessed', required=False
    )
    parser.add_argument(
        '--raw-directory', type=str, help='A path to the folder (from the project root) with raw data',
        default='datasets_raw', required=False
    )
    parser.add_argument(
        '--plots-save-dir', type=str, help='A path to the folder (from the project root) to save plots',
        default='results', required=False
    )
    parser.add_argument(
        '--data-sr', type=int, help='A sampling rate of data to be loaded', default=1000, required=False
    )
    parser.add_argument(
        '--resample-sr', type=int, help='A sampling rate to resample to for calculations', default=10, required=False
    )
    parser.add_argument(
        '--stimulus-time-seconds', type=int, help='Stimulus time in seconds', default=1, required=False
    )
    parser.add_argument(
        '--prestimulus-time-start-seconds', type=float, help='Prestimulus start time in seconds', default=- 0.5,
        required=False
    )
    parser.add_argument(
        '--prestimulus-time-finish-seconds', type=float, help='Prestimulus finish time in seconds', default=0,
        required=False
    )
    parser.add_argument(
        '--freq-band-low', type=int, help='A lower bound of a frequency band', default=60, required=False
    )
    parser.add_argument(
        '--freq-band-high', type=int, help='A higher bound of a frequency band', default=100, required=False
    )
    parser.add_argument(
        '--downsample', type=str, help='Downsample type', choices=['resample', 'average'], default='average',
        required=False
    )
    parser.add_argument(
        '--hilbert-use', action=argparse.BooleanOptionalAction,
        help='If True, applies Hilbert transformation to data', default=False
    )
    parser.add_argument(
        '--power-use', action=argparse.BooleanOptionalAction,
        help='If True, raises data to a second power', default=True
    )
    parser.add_argument(
        '--baseline-type', type=str, help='A type of a baseline', choices=[
            'absolute baseline'
            'z-score baseline'
            'relative baseline'
            'log-transform baseline'
            'absolute-relative baseline'
            'normed baseline'
            'log-averaged baseline'
        ],
        default='log-averaged baseline', required=False
    )
    parser.add_argument(
        '--alignment-type', type=str, help='A type of an alignment', choices=[
            'stimulus'
            'voice'
            'stimulus-voice'
        ],
        default='stimulus-voice', required=False
    )
    parser.add_argument(
        '--thresholds', type=float, nargs='+', help='Thresholds for confusion matrix calculation',
        required=False, default=[0.3, 0.4, 0.5]
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    models = [
        ('xgboost', None, None),
        ('svc', 'linear', 0.01),
        ('logreg', 'l2', 0.1),
        ('logreg', 'l1', 0.1),
    ]

    run_monopolar_experiments(
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
        root_dir=PROJECT_ROOT,
        raw_directory=args.raw_directory
    )
