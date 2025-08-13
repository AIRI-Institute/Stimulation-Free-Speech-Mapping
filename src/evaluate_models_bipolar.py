import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, balanced_accuracy_score, precision_recall_curve, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

from src.constants import PROJECT_ROOT
from src.evaluate_models import parse_arguments, plot_results, run_trial


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


def run_classification(model_type, features_train, features_test, df_fit, subject_names, save_path):
    df_test_results_init = df_fit.copy()
    df_test_results_init['prediction_proba'] = 0.0
    df_test_results = [df_test_results_init.copy() for _ in range(features_test.shape[0])]
    per_subject_grid_search = {}

    for index, subject_name in enumerate(subject_names):
        # Prepare the data
        df_train = df_fit.copy()
        df_train = df_train[df_train['subject_name'] != subject_name]

        df_test = df_fit.copy()
        df_test = df_test[df_test['subject_name'] == subject_name]

        X_train = features_train[df_train.index.values]
        X_tests = features_test[:, df_test.index.values, :]

        y_train = df_train.label.values

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        for i in range(X_tests.shape[0]):
            X_tests[i] = scaler.transform(X_tests[i])

        # Define the model and its hyperparameter grid
        if model_type == 'xgboost':
            clf = XGBClassifier(max_depth=3, random_state=42, min_child_weight=1, subsample=1)
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.3],
                'gamma': [0, 2],
                'colsample_bytree': [0.75, 1]
            }
        elif model_type == 'svc':
            clf = SVC(kernel='linear', gamma='scale', probability=True, random_state=42)
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1]
            }
        elif model_type == 'logreg':
            clf = LogisticRegression(random_state=42, max_iter=2000)
            param_grid = [
                {'penalty': ['l1'], 'solver': ['liblinear'], 'C': [0.01, 0.1, 1, 10, 100]},
                {'penalty': ['l2'], 'solver': ['liblinear', 'lbfgs'], 'C': [0.01, 0.1, 1, 10, 100]},
                {'penalty': ['elasticnet'], 'solver': ['saga'], 'C': [0.01, 0.1, 1, 10, 100], 'l1_ratio': [0.1, 0.25, 0.5, 0.75, 0.9]},
            ]
        elif model_type == 'random_forest':
            clf = RandomForestClassifier(random_state=42)
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
            clf = DecisionTreeClassifier(random_state=42)
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
        elif model_type == 'naive_bayes':
            clf = GaussianNB()
            param_grid = {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            }
        else:
            raise NotImplementedError(f'Model {model_type} is not implemented')

        # Perform grid search if there are hyperparameters to tune
        if param_grid:
            grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_clf = grid_search.best_estimator_
            grid_search_results = pd.DataFrame(grid_search.cv_results_).sort_values(
                by='mean_test_score', ascending=False
            )
        else:
            best_clf = clf
            best_clf.fit(X_train, y_train)
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


def evaluate_multiple_models(
    freq_band, fs, sr, downsample, hilbert_use,
    power_use, baseline_type, alignment_type, stimulus_time_seconds, prestimulus_time_start_seconds,
    prestimulus_time_finish_seconds, models, epochs_ieeg_trial, df_epochs_electrodes, subject_names,
    save_path, thresholds, perc_trials, n_bootstraps, epochs_sound_trial=None,
    print_message='', disable_tqdm=False
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
        bipolar=True,
        perc_trials=perc_trials,
        n_bootstraps=n_bootstraps,
        sound=False
    )

    if epochs_sound_trial is not None:
        features_sound_train, features_sound_test, _, _ = run_trial(
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
            perc_trials=perc_trials,
            n_bootstraps=n_bootstraps,
            sound=True
        )
        features_train = np.concatenate([features_train, features_sound_train], axis=-1)
        features_test = np.concatenate([features_test, features_sound_test], axis=-1)

    print(print_message)

    results_all = [{} for _ in range(features_test.shape[0])]
    for model_type_ in tqdm(models, desc='Running classification for every model', disable=disable_tqdm):
        label = model_type_
        model_results_path = os.path.join(save_path, label)
        Path(model_results_path).mkdir(parents=True, exist_ok=True)

        df_test_results_all, per_subject_grid_search = run_classification(
            model_type=model_type_, features_train=features_train, features_test=features_test, df_fit=df_fit,
            subject_names=subject_names, save_path=model_results_path
        )
        for df_test_idx, df_test_results in enumerate(df_test_results_all):
            results_all[df_test_idx][label] = df_test_results

        model_results_path = os.path.join(model_results_path, 'grid_search')
        Path(model_results_path).mkdir(parents=True, exist_ok=True)
        for subject_name, gs_results in per_subject_grid_search.items():
            if gs_results is not None:
                gs_results.to_csv(os.path.join(model_results_path, f'grid_search_sub_{subject_name}.csv'), index=False)

    rocs, prs = [[] for _ in range(features_test.shape[0])], [[] for _ in range(features_test.shape[0])]

    for df_test_idx, results in enumerate(results_all):

        for label, result in results.items():
            df_test_results = results[label].loc[df_fit.index]

            y_true = df_test_results['label'].values
            y_score = df_test_results['prediction_proba'].values

            fpr, tpr, thresholds_ftpr = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            rocs[df_test_idx].append((label, roc_auc, fpr, tpr))

            precision, recall, thresholds_pr = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall, precision)

            prs[df_test_idx].append((label, pr_auc, recall, precision))

            balanced_accuracy = balanced_accuracy_score(y_true, y_score > 0.5)
            print(label.ljust(20, '.'), f'test {df_test_idx}.', balanced_accuracy)

        for threshold in thresholds:
            for label, df_test_results in results.items():
                metrics_per_subject = create_results_table(results_df=df_test_results, threshold=threshold)
                metrics_per_subject.to_csv(os.path.join(save_path, label,
                                                        f'results_{df_test_idx}_{label}_thres{threshold}.csv'))

    return results_all, rocs, prs


def run_bipolar_experiments(
    models, augmentations, freq_band, fs, sr, downsample, hilbert_use,
    power_use, baseline_type, alignment_type, stimulus_time_seconds, prestimulus_time_start_seconds,
    prestimulus_time_finish_seconds, thresholds, dirpath_datasets, plots_save_dir, root_dir, 
    perc_trials, n_bootstraps, disable_tqdm
):

    if n_bootstraps is None:
        plots_save_dir = os.path.join(root_dir, plots_save_dir, f'perc-test-trials-{perc_trials}',
                                      f'freqs-{freq_band[0]}-{freq_band[1]}', 'bipolar')
    else:
        plots_save_dir = os.path.join(root_dir, plots_save_dir, f'perc-test-trials-{perc_trials}-bootstrap',
                                      f'freqs-{freq_band[0]}-{freq_band[1]}', 'bipolar')

    if Path(plots_save_dir).exists():
        print(f'{plots_save_dir} already exists, skipping')
        return None

    dir_name_w_sound = 'with_sound'
    dir_name_wo_sound = 'without_sound'
    plots_save_dir_w_sound = os.path.join(plots_save_dir, dir_name_w_sound)
    plots_save_dir_wo_sound = os.path.join(plots_save_dir, dir_name_wo_sound)
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
    for i in tqdm(range(epochs_sound.shape[0]), disable=disable_tqdm):
        index = df_epochs_electrodes_bipolar[df_epochs_electrodes_bipolar.epoch_index_total == i].index.values
        epochs_sound_trial[index, :] = epochs_sound[i].reshape((1, -1))

    todo_experiments = []

    if 'with_sound' in augmentations:

        objects_with_sound = dict(
            save_path=plots_save_dir_w_sound,
            epochs_sound_trial=epochs_sound_trial,
            print_message='Bipolar, with sound',
            plots_name='bipolar_objects_with_sound'
        )
        todo_experiments.append(objects_with_sound)

    if 'without_sound' in augmentations:
        objects_without_sound = dict(
            save_path=plots_save_dir_wo_sound,
            epochs_sound_trial=None,
            print_message='Bipolar, without sound',
            plots_name='bipolar_objects_wo_sound'
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
            models=models, epochs_ieeg_trial=epochs_ieeg_trial, df_epochs_electrodes=df_epochs_electrodes_bipolar,
            subject_names=subject_names,
            save_path=params['save_path'],
            thresholds=thresholds,
            perc_trials=perc_trials,
            n_bootstraps=n_bootstraps,
            epochs_sound_trial=params['epochs_sound_trial'],
            print_message=params['print_message'],
            disable_tqdm=disable_tqdm
        )

        for idx in range(len(rocs)):
            plot_results(
                rocs=rocs[idx], prs=prs[idx], test_idx=idx,
                save_path=os.path.join(params['save_path'], f'{params["plots_name"]}_{idx}.png')
            )

        print('\n')

    return 0


def run_bipolar_experiments_multi_freq_band(
    models, augmentations, low_freq, high_freq, step_freq, band_width, fs, sr, downsample, hilbert_use,
    power_use, baseline_type, alignment_type, stimulus_time_seconds, prestimulus_time_start_seconds,
    prestimulus_time_finish_seconds, thresholds, dirpath_datasets, plots_save_dir, root_dir,
    perc_trials, n_bootstraps, n_workers
):
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

    results = Parallel(n_jobs=n_workers)(delayed(run_bipolar_experiments)(
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
        perc_trials=perc_trials,
        n_bootstraps=n_bootstraps,
        disable_tqdm=disable_tqdm
    ) for freq_band in freq_pairs)


if __name__ == '__main__':
    args = parse_arguments()

    run_bipolar_experiments_multi_freq_band(
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
        perc_trials=args.perc_trials,
        n_bootstraps=args.n_bootstraps,
        n_workers=args.n_workers
    )
