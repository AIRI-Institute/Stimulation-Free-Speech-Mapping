import argparse
import json
import os
from pathlib import Path
from typing import Optional

import einops
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
from joblib import delayed, Parallel
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import compute_sample_weight
from tqdm import tqdm
from xgboost import XGBClassifier

from constants import ESM_NEGATIVE, ESM_POSITIVE, PROJECT_ROOT
from validation import get_average_roc, get_statistic, get_tables_subject


def load_json(json_path: str, encoding: Optional[str] = None):
    """
    Loads a json file into a dict

    Parameters
    ----------
    json_path : str
        A path to the json file
    encoding : str or None
        A specific encoding to use while reading json. If None, uses default encoder. Default: None

    Returns
    -------
    jsf : dict
        json file as a dict
    """
    if encoding is None:
        with open(json_path) as f:
            jsf = json.load(f)
    else:
        with open(json_path, encoding=encoding) as f:
            jsf = json.load(f)
    return jsf


def save_json(save_path: str, data: dict):
    """
    Save a dict into a json file

    Parameters
    ----------
    save_path : str
        A path to save file
    data : dict
        A dict with data to be saved
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


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


def logpower_results(epochs, df, bipolar, perc_trials=None, n_bootstraps=None):
    if perc_trials is not None:
        assert 0 < perc_trials <= 1, perc_trials
    df_results = create_df_results(df, bipolar=bipolar)
    results_train = np.full((df_results.shape[0], epochs.shape[-1]), fill_value=np.inf)
    if n_bootstraps is None:
        results_test = np.full((1, df_results.shape[0], epochs.shape[-1]), fill_value=np.inf)
    else:
        assert isinstance(n_bootstraps, int) and (n_bootstraps > 0), n_bootstraps
        results_test = np.full((n_bootstraps, df_results.shape[0], epochs.shape[-1]), fill_value=np.inf)
    for i, row in df_results.iterrows():
        indices = row['indices']
        epochs_channel_mean = epochs[indices].mean(axis=0)
        results_train[i] = epochs_channel_mean
        if perc_trials is not None:
            n_trials = round(perc_trials * len(indices))
            if n_bootstraps is None:
                first_n_indices = np.sort(indices)[:n_trials]
                results_test[0, i, :] = epochs[first_n_indices].mean(axis=0)
            else:
                for b_idx in range(n_bootstraps):
                    rng = np.random.default_rng(i * n_bootstraps + b_idx)
                    bootstrap_indices = np.sort(rng.choice(indices, size=n_trials, replace=True))
                    results_test[b_idx, i, :] = epochs[bootstrap_indices].mean(axis=0)
        else:
            if n_bootstraps is None:
                results_test[0, i, :] = epochs_channel_mean
            else:
                for b_idx in range(n_bootstraps):
                    rng = np.random.default_rng(i * n_bootstraps + b_idx)
                    bootstrap_indices = np.sort(rng.choice(indices, size=len(indices), replace=True))
                    results_test[b_idx, i, :] = epochs[bootstrap_indices].mean(axis=0)
    assert not np.isinf(results_train).any()
    assert not np.isinf(results_test).any()
    return results_train, results_test, df_results


def run_trial(
        epochs_ieeg, df_epochs_electrodes, freq_band, fs, sr, downsample, hilbert_use,
        power_use, baseline_type, alignment_type, stimulus_time_seconds, prestimulus_time_start_seconds,
        prestimulus_time_finish_seconds, bipolar, perc_trials, n_bootstraps, sound=False
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
    features_train, features_test, _ = logpower_results(epochs, df_ieeg, bipolar=bipolar, perc_trials=perc_trials,
                                                        n_bootstraps=n_bootstraps)

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

    return features_train, features_test, df_fit, df_results


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


def run_classification(model_type, features_train, features_test, df_fit, df_results, subject_names, save_path, 
                       class_weight):
    df_test_results = [df_results.copy() for _ in range(features_test.shape[0])]
    per_subject_grid_search = {}

    for idx, subject_name in enumerate(subject_names):
        # Prepare the data
        df_train = df_fit.copy()
        df_train = df_train[df_train['subject_name'] != subject_name]

        df_test = df_results.copy()
        df_test = df_test[df_test['subject_name'] == subject_name]
        df_test = df_test.drop(
            columns=['esm_type', 'esm_result_info', 'esm_result_concensus', 'indices', 'esm_result_info'])

        X_train = features_train[df_train.index.values]
        X_tests = features_test[:, df_test.index.values, :]

        y_train = df_train.label.values

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        for i in range(X_tests.shape[0]):
            X_tests[i] = scaler.transform(X_tests[i])

        sample_weight = None
        # Define the model and its hyperparameter grid
        if model_type == 'xgboost':
            if class_weight:
                n_pos_values = np.sum(y_train)
                scale_pos_weight = (y_train.shape[0] - n_pos_values) / n_pos_values
            else:
                scale_pos_weight = 1
            clf = XGBClassifier(
                max_depth=3, random_state=42, min_child_weight=1, subsample=1
            )
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.3],
                'gamma': [0, 2],
                'colsample_bytree': [0.75, 1],
                'scale_pos_weight': [scale_pos_weight]
            }
        elif model_type == 'svc':
            clf = SVC(kernel='linear', gamma='scale', probability=True, random_state=42, class_weight=class_weight)
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1]
            }
        elif model_type == 'logreg':
            clf = LogisticRegression(random_state=42, max_iter=2000, class_weight=class_weight)
            param_grid = [
                {'penalty': ['l1'], 'solver': ['liblinear'], 'C': [0.01, 0.1, 1, 10, 100]},
                {'penalty': ['l2'], 'solver': ['liblinear', 'lbfgs'], 'C': [0.01, 0.1, 1, 10, 100]},
                {'penalty': ['elasticnet'], 'solver': ['saga'], 'C': [0.01, 0.1, 1, 10, 100], 'l1_ratio': [0.1, 0.25, 0.5, 0.75, 0.9]},
            ]
        elif model_type == 'random_forest':
            clf = RandomForestClassifier(random_state=42, class_weight=class_weight)
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [None, 10],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4],
                'bootstrap': [True, False]
            }
        elif model_type == 'knn':
            clf = KNeighborsClassifier()
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        elif model_type == 'decision_tree':
            clf = DecisionTreeClassifier(random_state=42, class_weight=class_weight)
            param_grid = {
                'max_depth': [None, 10, 20, 30, 40],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
        elif model_type == 'mlp':
            clf = MLPClassifier(activation='relu', learning_rate='adaptive', max_iter=1000, random_state=42)
            param_grid = {
                'hidden_layer_sizes': [(100,), (100, 100)],
                'alpha': [0.0001, 0.001, 0.01]
            }
        elif model_type == 'adaboost':
            clf = AdaBoostClassifier(random_state=42, algorithm='SAMME')
            param_grid = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.05, 0.1, 0.2]
            }
            if class_weight:
                sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
        elif model_type == 'naive_bayes':
            clf = GaussianNB()
            param_grid = {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            }
            if class_weight:
                sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
        else:
            raise NotImplementedError(f'Model {model_type} is not implemented')

        # Perform grid search if there are hyperparameters to tune
        if param_grid:
            grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            if model_type in ('knn', 'mlp'):
                grid_search.fit(X_train, y_train)
            else:
                grid_search.fit(X_train, y_train, sample_weight=sample_weight)
            best_clf = grid_search.best_estimator_
            grid_search_results = pd.DataFrame(grid_search.cv_results_).sort_values(
                by='mean_test_score', ascending=False
            )
        else:
            best_clf = clf
            if model_type in ('knn', 'mlp'):
                best_clf.fit(X_train, y_train)
            else:    
                best_clf.fit(X_train, y_train, sample_weight=sample_weight)
            grid_search_results = None

        if model_type in {'svc', 'logreg'}:
            if model_type == 'svc':
                weights = dict(
                    coef=best_clf.coef_,
                    dual_coef=best_clf.dual_coef_,
                    intercept=best_clf.intercept_,
                    support=best_clf.support_,
                    support_vectors_=best_clf.support_vectors_,
                    n_support=best_clf.n_support_,
                    probA=best_clf.probA_,
                    probB=best_clf.probB_,
                    class_weight=best_clf.class_weight_
                )
            elif model_type == 'logreg':
                weights = dict(
                    coef=best_clf.coef_,
                    intercept=best_clf.intercept_
                )
            model_dir = os.path.join(save_path, 'model_weights')
            data_dir = os.path.join(save_path, 'data')
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            np.savez(os.path.join(model_dir, f'model_sub_{subject_name}.npz'), **weights)
            np.savez(
                os.path.join(data_dir, f'data_sub_{subject_name}.npz'),
                X_train=X_train, y_train=y_train, X_tests=X_tests
            )

        for b_idx in range(X_tests.shape[0]):
            # Make predictions
            y_hat_test = best_clf.predict_proba(X_tests[b_idx])[:, -1]

            # Save predictions
            df_test_results[b_idx].loc[df_test_results[b_idx]['subject_name'] == subject_name, 'prediction_proba'] = y_hat_test
        per_subject_grid_search[subject_name] = grid_search_results
    return df_test_results, per_subject_grid_search


def get_all_metrics(df_test_idx, df_test_results, label, subject_names, dfs_active, results_dir):
    esm_negative_ = set(str(esm_value) for esm_value in ESM_NEGATIVE)
    esm_positive_ = set(str(esm_value) for esm_value in ESM_POSITIVE)
    thresholds = np.concatenate([[0], np.sort(df_test_results['prediction_proba'].values), [1]])
    print('Prediction values:', df_test_results['prediction_proba'].values)
    print('Thresholds:', thresholds)
    tables = get_tables_subject(df_test_results, thresholds, subject_names, dfs_active, esm_negative_,
                                esm_positive_)
    
    fpr, tpr = get_average_roc(tables, thresholds, subject_names)
    roc_auc = get_roc_auc(fpr, tpr)
    rec, pr = get_average_pr(tables, thresholds, subject_names)
    pc_auc = get_pc_auc(rec, pr)
    senc, spec = get_average_ss(tables, thresholds, subject_names)
    ss_auc = get_ss_auc(senc, spec)

    print('TPR:', tpr)
    print('FPR:', fpr)
    print(f'{label}, test {df_test_idx}, roc_auc {roc_auc}, pc_auc {pc_auc}, ss_auc {ss_auc}')

    tables_dir = os.path.join(results_dir, 'confusion_matrices')
    Path(tables_dir).mkdir(parents=True, exist_ok=True)
    thresholds_list = thresholds.tolist()
    for subject_name, subject_table in tables.items():
        jsf = {}
        for thresh_idx, thresh in enumerate(thresholds_list):
            jsf[thresh] = subject_table[thresh_idx]
        save_json(save_path=os.path.join(tables_dir, f'cm_{subject_name}_{df_test_idx}.json'), data=jsf)

    return df_test_idx, label, roc_auc, fpr, tpr, pc_auc, rec, pr, df_test_results


def evaluate_multiple_models(
        freq_band, fs, sr, downsample, hilbert_use,
        power_use, baseline_type, alignment_type, stimulus_time_seconds, prestimulus_time_start_seconds,
        prestimulus_time_finish_seconds, models, epochs_ieeg_trial, df_epochs_electrodes, subject_names, dfs_active,
        save_path, thresholds_cm, perc_trials, n_bootstraps, class_weight, epochs_sound_trial=None,
        print_message='', n_workers=None
):
    features_train, features_test, df_fit, df_results = run_trial(
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
        perc_trials=perc_trials,
        n_bootstraps=n_bootstraps,
        sound=False
    )

    if epochs_sound_trial is not None:
        features_sound_train, features_sound_test, df_fit_, df_results_ = run_trial(
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
            perc_trials=perc_trials,
            n_bootstraps=n_bootstraps,
            sound=False
        )
        features_train = np.concatenate([features_train, features_sound_train], axis=-1)
        features_test = np.concatenate([features_test, features_sound_test], axis=-1)

    print(print_message)

    results_all = [{} for _ in range(features_test.shape[0])]

    rocs, prs = [[] for _ in range(features_test.shape[0])], [[] for _ in range(features_test.shape[0])]
    
    for subject_name in subject_names:
        dfs_active[subject_name].to_csv(
            os.path.join(save_path, f'dfs_active_{subject_name}.csv'), index=False
        )
        

    for model_type_ in models:
        label = model_type_
        model_results_path = os.path.join(save_path, label)
        Path(model_results_path).mkdir(parents=True, exist_ok=True)

        df_test_results_all, per_subject_grid_search = run_classification(
            model_type=model_type_, features_train=features_train, features_test=features_test, df_fit=df_fit,
            df_results=df_results, subject_names=subject_names, save_path=model_results_path,
            class_weight=class_weight
        )

        model_full_results_path = os.path.join(model_results_path, 'grid_search')
        Path(model_full_results_path).mkdir(parents=True, exist_ok=True)
        for df_test_idx, df_test_results in enumerate(df_test_results_all):
            df_test_results.to_csv(os.path.join(model_full_results_path, f'test_results_{df_test_idx}.csv'), index=False)

        if n_workers is None:
            metrics_results = []

            for df_test_idx, df_test_results in enumerate(df_test_results_all):

                metrics_results.append(get_all_metrics(
                    df_test_idx=df_test_idx, df_test_results=df_test_results, label=label,
                    subject_names=subject_names, dfs_active=dfs_active,
                    results_dir=model_results_path
                ))

        else:
            metrics_results = Parallel(n_jobs=n_workers)(delayed(get_all_metrics)(
                df_test_idx=df_test_idx, df_test_results=df_test_results, label=label,
                subject_names=subject_names, dfs_active=dfs_active, 
                results_dir=model_results_path
            ) for df_test_idx, df_test_results in enumerate(df_test_results_all))

        for df_test_idx, label, roc_auc, fpr, tpr, pc_auc, rec, pr, df_test_results in metrics_results:
            rocs[df_test_idx].append((label, roc_auc, fpr, tpr))
            prs[df_test_idx].append((label, pc_auc, rec, pr))

            results_all[df_test_idx][label] = df_test_results

        model_results_path = os.path.join(model_results_path, 'grid_search')
        Path(model_results_path).mkdir(parents=True, exist_ok=True)
        for subject_name, gs_results in per_subject_grid_search.items():
            if gs_results is not None:
                gs_results.to_csv(os.path.join(model_results_path, f'grid_search_sub_{subject_name}.csv'), index=False)

    for df_test_idx, results in enumerate(results_all):
        for label, df_test_results in results.items():
            esm_negative_ = set(str(esm_value) for esm_value in ESM_NEGATIVE)
            esm_positive_ = set(str(esm_value) for esm_value in ESM_POSITIVE)
            tables = get_tables_subject(df_test_results, thresholds_cm, subject_names, dfs_active, esm_negative_,
                                        esm_positive_)
            results_tables = create_results_table(per_subject_tables=tables, thresholds=thresholds_cm)
            for threshold, metrics_per_subject in results_tables.items():
                metrics_per_subject.to_csv(
                    os.path.join(save_path, label, f'results_{df_test_idx}_{label}_thres{threshold}.csv')
                )

    return results_all, rocs, prs


def plot_results(rocs, prs, test_idx, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    xlabels = ['False Positive Rate', 'Recall']
    ylabels = ['True Positive Rate', 'Precision']
    titles = ['ROC Curve', 'PR Curve']
    locs = ['lower right', 'lower left']
    labels = {
        'xgboost': 'XGBoost',
        'svc': 'SVC',
        'logreg': 'LR',  # 'Logistic Regression',
        'random_forest': 'RF',  #  'Random Forest',
        'knn': 'KNN',
        'decision_tree': 'DT',  # 'Decision Tree',
        'mlp': 'MLP',
        'adaboost': 'AdaBoost',
        'naive_bayes': 'NB'  #  'Gaussian Naive Bayes'
    }
    overall_table = {}

    print(f'ROC, test {test_idx}')
    for label, roc_auc, fpr, tpr, in rocs:
        ax[0].plot(fpr, tpr, label=labels[label])
        print(labels[label].ljust(40, '.') + str(round(roc_auc, 4)))
        assert label not in overall_table, (overall_table, label)
        overall_table[label] = {'ROC': round(roc_auc, 4)}

    print(f'PR, test {test_idx}')
    for label, pc_auc, recall, precision in prs:
        ax[1].plot(recall, precision, label=labels[label])
        print(labels[label].ljust(40, '.') + str(round(pc_auc, 4)))
        overall_table[label]['PR'] = round(pc_auc, 4)

    results_table = {'Model': [], 'ROC': [], 'PR': []}
    for label, res in overall_table.items():
        results_table['Model'].append(labels[label])
        results_table['ROC'].append(res['ROC'])
        results_table['PR'].append(res['PR'])
    results_table = pd.DataFrame(results_table)
    results_table.to_csv(os.path.splitext(save_path)[0] + '.csv', index=False)

    for i in range(2):
        ax[i].set_xlim(0, 1)
        ax[i].set_ylim(0, 1)
        # ax[i].legend(loc=locs[i])
        ax[i].legend(ncol=2, labelspacing=0.05)  # ,handleheight=2.4)
        ax[i].set_xlabel(xlabels[i])
        ax[i].set_ylabel(ylabels[i])
        ax[i].set_title(titles[i])

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    return 0


def run_monopolar_experiments(
    models, augmentations, freq_band, fs, sr, downsample, hilbert_use,
    power_use, baseline_type, alignment_type, stimulus_time_seconds, prestimulus_time_start_seconds,
    prestimulus_time_finish_seconds, thresholds, dirpath_datasets, plots_save_dir, root_dir, raw_directory,
    perc_trials, n_bootstraps, disable_tqdm, n_workers, class_weight
):

    if n_bootstraps is None:
        plots_save_dir = os.path.join(root_dir, plots_save_dir, f'perc-test-trials-{perc_trials}',
                                      f'freqs-{freq_band[0]}-{freq_band[1]}', 'monopolar')
    else:
        plots_save_dir = os.path.join(root_dir, plots_save_dir, f'perc-test-trials-{perc_trials}-bootstrap',
                                      f'freqs-{freq_band[0]}-{freq_band[1]}', 'monopolar')

    if Path(plots_save_dir).exists():
        print(f'{plots_save_dir} already exists, skipping')
        return None

    dir_name_w_sound = 'with_sound'
    dir_name_wo_sound = 'without_sound'
    plots_save_dir_w_sound = os.path.join(plots_save_dir, dir_name_w_sound)
    plots_save_dir_wo_sound = os.path.join(plots_save_dir, dir_name_wo_sound)
    os.makedirs(plots_save_dir_w_sound, exist_ok=True)
    os.makedirs(plots_save_dir_wo_sound, exist_ok=True)

    epochs_ieeg_trial = np.load(os.path.join(root_dir, dirpath_datasets, f'epochs_ieeg{fs}_float32.npy'))
    epochs_sound = np.load(os.path.join(root_dir, dirpath_datasets, f'epochs_sound{fs}_float32.npy'))
    df_epochs_electrodes = pd.read_csv(
        os.path.join(root_dir, dirpath_datasets, 'df_epochs_electrodes.csv'), index_col=0
    )

    subject_names = df_epochs_electrodes['subject_name'].unique()
    dfs_active = {}
    for subject_name in tqdm(subject_names, desc='Loading subjects data', disable=disable_tqdm):
        subject_info = os.path.join(root_dir, raw_directory, 'sheets', 'subjects_sheets', f'{subject_name}.xlsx')
        dfs_active[subject_name] = pd.read_excel(subject_info, sheet_name='Active Stimulation')[
            ['Channel 1', 'Channel 2', 'Result']]

    df_epochs_electrodes = pd.read_csv(
        os.path.join(root_dir, dirpath_datasets, 'df_epochs_electrodes.csv'), index_col=0
    )

    df_epochs_electrodes.loc[pd.isna(df_epochs_electrodes['voice_start1000'])] = -1
    df_epochs_electrodes['voice_start1000'] = df_epochs_electrodes['voice_start1000'].astype(int)

    epochs_sound_trial = np.zeros((df_epochs_electrodes.shape[0], epochs_sound.shape[1]))
    for i in tqdm(range(epochs_sound.shape[0]), desc='Collecting epochs for sounds', disable=disable_tqdm):
        epoch_indices = df_epochs_electrodes[df_epochs_electrodes.epoch_index_total == i].index.values
        epochs_sound_trial[epoch_indices, :] = epochs_sound[i].reshape((1, -1))

    todo_experiments = []

    if 'with_sound' in augmentations:
        objects_with_sound = dict(
            save_path=plots_save_dir_w_sound,
            epochs_sound_trial=epochs_sound_trial,
            print_message='Monopolar, with sound',
            plots_name='monopolar_objects_with_sound'
        )
        todo_experiments.append(objects_with_sound)

    if 'without_sound' in augmentations:
        objects_without_sound = dict(
            save_path=plots_save_dir_wo_sound,
            epochs_sound_trial=None,
            print_message='Monopolar, without sound',
            plots_name='monopolar_objects_wo_sound'
        )
        todo_experiments.append(objects_without_sound)

    for params in todo_experiments:

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
            perc_trials=perc_trials,
            n_bootstraps=n_bootstraps,
            class_weight=class_weight,
            epochs_sound_trial=params['epochs_sound_trial'],
            print_message=params['print_message'],
            n_workers=n_workers
        )

        for idx in range(len(rocs)):
            plot_results(
                rocs=rocs[idx], prs=prs[idx], test_idx=idx,
                save_path=os.path.join(params['save_path'], f'{params["plots_name"]}_{idx}.png')
            )

        print('\n')

    return 0


def run_monopolar_experiments_multi_freq_band(
    models, augmentations, low_freq, high_freq, step_freq, band_width, fs, sr, downsample, hilbert_use,
    power_use, baseline_type, alignment_type, stimulus_time_seconds, prestimulus_time_start_seconds,
    prestimulus_time_finish_seconds, thresholds, dirpath_datasets, plots_save_dir, root_dir, raw_directory,
    perc_trials, n_bootstraps, n_workers, class_weight
):
    if class_weight:
        class_weight = 'balanced'
    else:
        class_weight = None
    assert low_freq > 0, low_freq
    assert high_freq > low_freq, (low_freq, high_freq)
    assert step_freq > 0, step_freq
    assert isinstance(low_freq, int), (type(low_freq), low_freq)
    assert isinstance(high_freq, int), (type(high_freq), high_freq)
    assert isinstance(step_freq, int), (type(step_freq), step_freq)

    if n_workers > 1:
        disable_tqdm = True
    else:
        disable_tqdm = False

    freqs = list(range(low_freq, high_freq + 1, step_freq))
    freq_pairs = []
    assert band_width > 0, band_width
    assert isinstance(band_width, int), (type(band_width), band_width)
    half_width = band_width // 2
    for c_freq1 in freqs:
        for c_freq2 in freqs:
            if c_freq1 > c_freq2:
                continue
            freq_band = (c_freq1 - half_width, c_freq2 + half_width)
            if freq_band not in freq_pairs:
                freq_pairs.append(freq_band)

    if (n_bootstraps is None) or (n_bootstraps == 1) or (n_bootstraps < len(freq_pairs)):

        results = Parallel(n_jobs=n_workers)(delayed(run_monopolar_experiments)(
            models=models,
            augmentations=augmentations,
            freq_band=np.asarray(list(freq_band)),
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
            thresholds=thresholds,
            dirpath_datasets=dirpath_datasets,
            plots_save_dir=plots_save_dir,
            root_dir=root_dir,
            raw_directory=raw_directory,
            perc_trials=perc_trials,
            n_bootstraps=n_bootstraps,
            disable_tqdm=disable_tqdm,
            n_workers=None,
            class_weight=class_weight
        ) for freq_band in freq_pairs)

    else:
        results = []

        for freq_band in freq_pairs:
            res = run_monopolar_experiments(
                models=models,
                augmentations=augmentations,
                freq_band=np.asarray(list(freq_band)),
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
                thresholds=thresholds,
                dirpath_datasets=dirpath_datasets,
                plots_save_dir=plots_save_dir,
                root_dir=root_dir,
                raw_directory=raw_directory,
                perc_trials=perc_trials,
                n_bootstraps=n_bootstraps,
                disable_tqdm=disable_tqdm,
                n_workers=n_workers
            )
            results.append(res)


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
        '--augmentations', type=str, nargs='+', help='Augmentations to plot for',
        choices=['with_sound', 'without_sound'], default=('with_sound', 'without_sound'), required=False
    )
    parser.add_argument(
        '--models', type=str, nargs='+', help='Models to use',
        choices=['xgboost', 'svc', 'logreg', 'random_forest', 'knn', 'decision_tree', 'mlp', 'adaboost', 'naive_bayes'],
        default=['xgboost', 'adaboost', 'svc', 'logreg', 'random_forest', 'knn', 'decision_tree', 'mlp', 'naive_bayes'],
        required=False
    )
    parser.add_argument(
        '--low-freq', type=int, help='A lower bound of a potential central frequency', default=125, required=False
    )
    parser.add_argument(
        '--high-freq', type=int, help='A higher bound of a potential central frequency', default=126, required=False
    )
    parser.add_argument(
        '--step-freq', type=int, help='A step in frequencies between consecutive central frequencies', default=10,
        required=False
    )
    parser.add_argument(
        '--band-width', type=int, help='A width of frequency bands', default=50, required=False
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
        required=False, default=[0.2, ]
    )
    parser.add_argument(
        '--perc-trials', type=float, help='Percentage of trials to use in test. If None, uses all data', required=False, default=None
    )
    parser.add_argument(
        '--n-bootstraps', type=int, help='Number of bootstraps to use. If None, no bootstrap is applied',
        required=False, default=None
    )
    parser.add_argument(
        '--n-workers', type=int, help='Number of workers to use.', required=False, default=1
    )
    parser.add_argument(
        '--class-weight', action=argparse.BooleanOptionalAction,
        help='If True, classification algorithms will utilize class weighting during fit', default=False
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    plt.rcParams.update({'font.size': 16})

    run_monopolar_experiments_multi_freq_band(
        models=args.models,
        augmentations=args.augmentations,
        low_freq=args.low_freq,
        high_freq=args.high_freq,
        step_freq=args.step_freq,
        band_width=args.band_width,
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
        raw_directory=args.raw_directory,
        perc_trials=args.perc_trials,
        n_bootstraps=args.n_bootstraps,
        n_workers=args.n_workers,
        class_weight=args.class_weight
    )
