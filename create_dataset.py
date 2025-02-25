import argparse
import os
from copy import deepcopy

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from constants import PROJECT_ROOT, SAMPLING_RATE


class SimpleVAD:
    def __init__(self, fs, fsds=40, ignore_voice=1, ignore_silence=7):
        self.fsds = fsds
        self.fs = fs
        self.frame = int(fs / self.fsds)
        self.ignore_voice = ignore_voice
        self.ignore_silence = ignore_silence

    def detect_voice(self, sound, threshold=100):
        sound_frames = [sound[i:i + self.frame] for i in range(0, sound.shape[0], self.frame)]
        energy_frames = np.asarray([np.mean(s ** 2) for s in sound_frames])
        voice_detected = (energy_frames >= threshold).astype(np.int32)

        counter_voice = 0
        counter_silence = 0
        voice_detected_smoothed = np.zeros(voice_detected.size).astype(np.int32)
        for i in range(voice_detected.shape[0]):
            if voice_detected[i] == 1:
                counter_voice += 1
                if counter_voice > self.ignore_voice:
                    voice_detected_smoothed[i - counter_voice + 1:i + 1] = 1
                    counter_silence = 0
            else:
                counter_silence += 1
                if counter_silence > self.ignore_silence:
                    voice_detected_smoothed[i - self.ignore_silence:i + 1] = 0
                    counter_voice = 0
        ignore_max = np.max([self.ignore_voice, self.ignore_silence])
        voice_detected_smoothed[:ignore_max] = 0
        voice_detected_smoothed[-ignore_max:] = 0

        return voice_detected_smoothed

    def voiced_index(self, sound, window, baseline_length, left_count, **kwargs):
        voice_detected = self.detect_voice(sound, kwargs['threshold'])
        baseline_frames = round(baseline_length * self.fsds)
        offset_frames_left = round(window[0] * self.fsds) + baseline_frames
        offset_frames_right = round(window[1] * self.fsds) + baseline_frames
        voiced_i = np.where(voice_detected)[0]

        if np.sum(np.where(np.logical_and(voiced_i > baseline_frames, voiced_i < offset_frames_left))[0]) > 3:
            for i in range(offset_frames_left, offset_frames_right):
                if voice_detected[i] == 1:
                    voice_detected[i] = 0
                else:
                    voiced_i = np.where(voice_detected)[0]
                    break

        voiced_left = voiced_i[np.where(np.logical_and(voiced_i >= baseline_frames, voiced_i < offset_frames_left))[0]]
        voiced_i = voiced_i[np.where(np.logical_and(voiced_i >= offset_frames_left, voiced_i < offset_frames_right))[0]]

        if np.sum(len(voiced_left)) > left_count:
            return None
        elif voiced_i.size == 0:
            return None
        else:
            return voiced_i[0] * self.frame

    def detect_multiple(self, epochs_sound, window, baseline_length, left_count, threshold, dirpath_datasets):

        voice_detected40 = []
        voice_indices = []

        for i in tqdm(range(epochs_sound.shape[0]), total=epochs_sound.shape[0], desc='Running VAD'):
            voice_detected = self.detect_voice(epochs_sound[i])
            voice_index = self.voiced_index(
                epochs_sound[i],
                window=window,
                baseline_length=baseline_length,
                left_count=left_count,
                threshold=threshold
            )
            voice_detected40.append(voice_detected)
            voice_indices.append(voice_index)

        voice_detected40 = np.stack(voice_detected40)
        voice_indices = np.asarray(voice_indices).astype(float)
        voice_valid = np.asarray([(0 if i is None else 1) for i in voice_indices])

        np.save(os.path.join(dirpath_datasets, 'voice_detected40.npy'), voice_detected40)
        np.save(os.path.join(dirpath_datasets, 'voice_indices.npy'), voice_indices)
        np.save(os.path.join(dirpath_datasets, 'voice_valid.npy'), voice_valid)

        return voice_detected40, voice_indices, voice_valid


def build_df_records_and_active(df_subjects, dirpath_subject_sheets, fif_directory):
    dfs_active = {}
    df_records = []

    for index, row in tqdm(df_subjects.iterrows(), total=df_subjects.shape[0], desc='Building df_records'):
        subject_name_ = str(row['subject_name'])
        filepath_subject_info = os.path.join(dirpath_subject_sheets, f'{subject_name_}.xlsx')
        df_subject = pd.read_excel(filepath_subject_info, sheet_name='Records')
        df_subject = df_subject[df_subject['Block'].notna()]
        df_subject_ = df_subject[df_subject['Block'].astype(int).astype(str).isin(row['blocks'].split(','))].copy()
        df_subject_['subject_name'] = subject_name_
        df_subject_['subject_id'] = row['subject_id']
        df_subject_['experiment_type'] = 'extra'
        df_subject_['file_name'] = df_subject_['File name']
        df_subject_['stimulus_type'] = df_subject_['Type']
        df_subject_['block_id'] = df_subject_['Block']
        df_records.append(df_subject_)

        df_active = pd.read_excel(filepath_subject_info, sheet_name='Active Stimulation')
        df_active = df_active[['Result', 'Channel 1', 'Channel 2']]
        dfs_active[subject_name_] = df_active

    df_records = pd.concat(df_records, ignore_index=True)
    df_records = df_records[['subject_name', 'subject_id', 'experiment_type', 'stimulus_type', 'block_id', 'file_name']]

    for i in range(df_records.shape[0]):
        file_name = df_records.loc[i]['file_name']
        subject_name = df_records.loc[i]['subject_name']
        path_file_fif = os.path.join(
            fif_directory, subject_name, file_name
        )
        if not os.path.isfile(path_file_fif):
            print(path_file_fif)

    return df_records, dfs_active


def build_df_epochs_electrodes(
        df_records, dfs_active, voice_indices, voice_valid, target_sr, baseline_duration_seconds,
        stimulus_duration_seconds, dirpath_datasets, fif_directory
):
    df_epochs_electrodes = []

    epoch_index_total = 0
    for i in tqdm(range(df_records.shape[0]), total=df_records.shape[0], desc='Building df_epochs_electrodes'):
        file_name = df_records.loc[i]['file_name']
        subject_name = df_records.loc[i]['subject_name']

        path_file_fif = os.path.join(
            fif_directory, subject_name, file_name
        )
        raw = mne.io.read_raw_fif(path_file_fif, preload=False, verbose=False)
        n_samples = raw.n_times

        assert SAMPLING_RATE == round(raw.info['sfreq'])

        df_active = dfs_active[subject_name]
        channel_indices = mne.pick_channels(raw.info['ch_names'], [], exclude=raw.info['bads'] + ['sound'])
        channel_names = np.asarray(raw.info['ch_names'])[channel_indices]
        esm_types, esm_results_concensus, esm_results_info = create_esm_result(channel_names, df_active)

        epoch_index = 0
        for annotation in raw.annotations:
            stimulus_onset = round(annotation['onset'] * SAMPLING_RATE)
            epoch_start = stimulus_onset - round(baseline_duration_seconds * SAMPLING_RATE)
            epoch_stop = stimulus_onset + round(stimulus_duration_seconds * SAMPLING_RATE)
            if epoch_start < 0:
                continue
            elif epoch_stop > n_samples:
                continue

            for index, name, esm_type, esm_result_concensus, esm_result_info in zip(channel_indices, channel_names,
                                                                                    esm_types, esm_results_concensus,
                                                                                    esm_results_info):
                new_row = df_records.loc[i].to_dict()
                new_row['channel_index'] = index
                new_row['channel_name'] = name
                new_row['esm_type'] = esm_type
                new_row['esm_result_concensus'] = esm_result_concensus
                new_row['esm_result_info'] = esm_result_info
                new_row['epoch_index'] = epoch_index
                new_row['epoch_index_total'] = epoch_index_total
                new_row[f'stimulus_onset_sr{SAMPLING_RATE}'] = stimulus_onset
                new_row[f'stimulus_onset_sr{target_sr}'] = round(
                    stimulus_onset / SAMPLING_RATE * target_sr)
                new_row['voice_indices'] = voice_indices[epoch_index_total]
                new_row['voice_valid'] = voice_valid[epoch_index_total]
                new_row['voice_start1000'] = voice_indices[epoch_index_total] if voice_valid[epoch_index_total] else 0
                new_row['voice_start500'] = voice_indices[epoch_index_total] // 2 if voice_valid[
                    epoch_index_total] else 0

                df_epochs_electrodes.append(new_row)

            epoch_index += 1
            epoch_index_total += 1

    df_epochs_electrodes = pd.DataFrame(df_epochs_electrodes)
    df_epochs_electrodes['voice_indices'] = df_epochs_electrodes['voice_indices'].astype('Int64')

    df_epochs_electrodes.to_csv(os.path.join(dirpath_datasets, 'df_epochs_electrodes.csv'))

    return df_epochs_electrodes


def get_epochs_sound(df_records, baseline_duration_seconds, stimulus_duration_seconds, fif_directory):
    epochs_sound4096 = []

    for i in tqdm(range(df_records.shape[0]), total=df_records.shape[0], desc='Extracting epochs_sound'):
        file_name = df_records.loc[i]['file_name']
        subject_name = df_records.loc[i]['subject_name']

        path_file_fif = os.path.join(
            fif_directory, subject_name, file_name
        )
        raw = mne.io.read_raw_fif(path_file_fif, preload=False, verbose=False)
        n_samples = raw.n_times

        sound = raw.pick(picks='sound').get_data()
        sound = (sound + 8982.25) / 15.25

        for annotation in raw.annotations:
            stimulus_onset = round(annotation['onset'] * SAMPLING_RATE)
            epoch_start = stimulus_onset - round(baseline_duration_seconds * SAMPLING_RATE)
            epoch_stop = stimulus_onset + round(stimulus_duration_seconds * SAMPLING_RATE)
            if epoch_start < 0:
                continue
            elif epoch_stop > n_samples:
                continue

            epoch_sound = sound[:, epoch_start:epoch_stop]
            epochs_sound4096.append(epoch_sound)

    epochs_sound4096 = np.concatenate(epochs_sound4096, axis=0)

    return epochs_sound4096


def create_esm_result(channel_names_, df_active_):
    active_negative = '0'
    active_positive_language = '1'
    active_positive_articulatory = '2'
    active_nonspeech_related = '3'
    active_known = [active_negative, active_positive_language, active_positive_articulatory, active_nonspeech_related]
    active_unknown = ['-', None]

    esm_info = np.zeros(channel_names_.shape[0], dtype=int)
    esm_types_ = -1 * np.ones(channel_names_.shape[0], dtype=int)
    esm_results_concensus_ = np.zeros(channel_names_.shape[0], dtype=int)
    esm_results_info_ = np.zeros(channel_names_.shape[0], dtype=int)

    for index, row in df_active_.iterrows():
        esm_result, channel_name1, channel_name2 = row
        for channel_name in (channel_name1, channel_name2):
            if channel_name in channel_names_:
                channel_index = channel_names_.tolist().index(channel_name)
                esm_info[channel_index] = 1

    for i, (info, channel_name) in enumerate(zip(esm_info, channel_names_)):
        if info:
            active_1, active_2 = None, None
            if channel_name in df_active_['Channel 1'].values:
                index_1 = df_active_['Channel 1'].values.tolist().index(channel_name)
                active_1 = str(df_active_.loc[index_1, 'Result'])
            if channel_name in df_active_['Channel 2'].values:
                index_2 = df_active_['Channel 2'].values.tolist().index(channel_name)
                active_2 = str(df_active_.loc[index_2, 'Result'])

            if active_1 == active_2:
                esm_results_concensus_[i] = 1
                esm_results_info_[i] = 1
                if active_1 in active_known:
                    esm_types_[i] = int(active_1)
                elif active_1 in active_unknown:
                    esm_types_[i] = -1

            elif active_1 != active_2:
                esm_results_info_[i] = 1
                if ((active_1 in active_known) and (active_2 in active_unknown)) or (
                        (active_2 in active_known) and (active_1 in active_unknown)):
                    if active_1 in active_known:
                        esm_types_[i] = int(active_1)
                    elif active_2 in active_known:
                        esm_types_[i] = int(active_2)

                elif (active_1 in active_known) and (active_2 in active_known):
                    if (active_1 == active_nonspeech_related) or (active_2 == active_nonspeech_related):
                        esm_types_[i] = int(active_nonspeech_related)
                    elif (active_1 == active_positive_articulatory) or (active_2 == active_positive_articulatory):
                        esm_types_[i] = int(active_positive_articulatory)
                    elif (active_1 == active_positive_language) or (active_2 == active_positive_language):
                        esm_types_[i] = int(active_positive_language)
                    elif (active_1 == active_negative) or (active_2 == active_negative):
                        esm_types_[i] = int(active_negative)
    return esm_types_, esm_results_concensus_, esm_results_info_


def get_ieeg_epochs(
        df_records, df_epochs_electrodes, target_sr, baseline_duration_seconds, stimulus_duration_seconds,
        dirpath_datasets, fif_directory
):
    # Minimal filtering - bandpass [0.5, 300], notch [50, 100, 150, 200, 250, 300, 350], CAR!

    offset = 0
    all_epochs_ieeg = np.zeros((df_epochs_electrodes.shape[0], round(
        (baseline_duration_seconds + stimulus_duration_seconds) * target_sr)), dtype=np.float32)

    for i in tqdm(range(df_records.shape[0]), total=df_records.shape[0], desc='Extracting ieeg_epochs'):
        file_name = df_records.loc[i]['file_name']
        subject_name = df_records.loc[i]['subject_name']

        path_file_fif = os.path.join(
            fif_directory, subject_name, file_name
        )
        raw = mne.io.read_raw_fif(path_file_fif, preload=True, verbose=False)
        n_samples = raw.n_times

        assert SAMPLING_RATE == round(raw.info['sfreq'])

        channel_indices = mne.pick_channels(raw.info['ch_names'], [], exclude=raw.info['bads'] + ['sound'])
        channel_names = np.asarray(raw.info['ch_names'])[channel_indices]

        raw.pick(picks='ecog', exclude='bads')
        for ch_i, channel in enumerate(raw.info['ch_names']):
            assert channel == channel_names[ch_i]

        raw.filter(l_freq=0.5, h_freq=300, verbose=False)
        raw.notch_filter(freqs=np.arange(50, 351, 50), verbose=False)
        raw.set_eeg_reference(ref_channels='average', projection=False, ch_type='ecog', verbose=False)

        ieeg = raw.get_data()

        for annotation in raw.annotations:
            stimulus_onset = round(annotation['onset'] * SAMPLING_RATE)
            epoch_start = stimulus_onset - round(baseline_duration_seconds * SAMPLING_RATE)
            epoch_stop = stimulus_onset + round(stimulus_duration_seconds * SAMPLING_RATE)
            if epoch_start < 0:
                continue
            elif epoch_stop > n_samples:
                continue

            epoch_ieeg = ieeg[:, epoch_start:epoch_stop]
            assert epoch_stop - epoch_start == SAMPLING_RATE * (
                    baseline_duration_seconds + stimulus_duration_seconds), (
                ieeg.shape, epoch_start, epoch_stop, epoch_stop - epoch_start)

            if target_sr == SAMPLING_RATE:
                resampled_epoch_ieeg = epoch_ieeg
            else:
                resampled_epoch_ieeg = mne.filter.resample(epoch_ieeg, down=SAMPLING_RATE / target_sr)

            all_epochs_ieeg[offset:offset + resampled_epoch_ieeg.shape[0]] = resampled_epoch_ieeg.astype(
                np.float32)
            offset += resampled_epoch_ieeg.shape[0]

    np.save(
        os.path.join(dirpath_datasets, f'epochs_ieeg{target_sr}_float32.npy'), all_epochs_ieeg
    )

    return all_epochs_ieeg


def get_bipolar_data(
        df_records, dfs_active, voice_indices, voice_valid, target_sr, baseline_duration_seconds,
        stimulus_duration_seconds, dirpath_datasets, fif_directory
):
    df_epochs_electrodes_bipolar = []
    epochs_ieeg_bipolar = []

    epoch_index_total = 0
    for i in tqdm(range(df_records.shape[0]), total=df_records.shape[0], desc='Extracting epochs_ieeg_bipolar'):
        file_name = df_records.loc[i]['file_name']
        subject_name = df_records.loc[i]['subject_name']

        path_file_fif = os.path.join(
            fif_directory, subject_name, file_name
        )
        raw = mne.io.read_raw_fif(path_file_fif, preload=True, verbose=False)
        n_samples = raw.n_times

        assert SAMPLING_RATE == round(raw.info['sfreq'])

        channel_indices = mne.pick_channels(raw.info['ch_names'], [], exclude=raw.info['bads'] + ['sound'])
        channel_names = np.asarray(raw.info['ch_names'])[channel_indices]

        df_active = dfs_active[subject_name]

        # EPOCHS
        epochs_indices = []
        for epoch_index, annotation in enumerate(raw.annotations):
            stimulus_onset = round(annotation['onset'] * SAMPLING_RATE)
            epoch_start = stimulus_onset - round(baseline_duration_seconds * SAMPLING_RATE)
            epoch_stop = stimulus_onset + round(stimulus_duration_seconds * SAMPLING_RATE)
            if epoch_start < 0:
                continue
            elif epoch_stop > n_samples:
                continue
            epochs_indices.append((epoch_index, epoch_index_total, epoch_start, stimulus_onset, epoch_stop))
            epoch_index_total += 1

        # RAW
        raw.filter(l_freq=0.5, h_freq=300, verbose=False)
        raw.notch_filter(freqs=np.arange(50, 351, 50), verbose=False)
        ieeg = raw.get_data()

        # CHANNELS
        for j in range(df_active.shape[0]):
            esm_type = df_active.loc[j]['Result']
            esm_channel_1 = df_active.loc[j]['Channel 1']
            esm_channel_2 = df_active.loc[j]['Channel 2']

            if esm_channel_1 not in channel_names or esm_channel_2 not in channel_names:
                continue
            esm_channel_1_index = channel_indices[np.where(channel_names == esm_channel_1)[0][0]]
            esm_channel_2_index = channel_indices[np.where(channel_names == esm_channel_2)[0][0]]
            ieeg_bipolar = ieeg[esm_channel_1_index] - ieeg[esm_channel_2_index]

            for epoch_index_, epoch_index_total_, epoch_start_, stimulus_onset_, epoch_stop_ in deepcopy(
                    epochs_indices):
                epoch_ieeg_bipolar = ieeg_bipolar[epoch_start_:epoch_stop_]
                if target_sr == SAMPLING_RATE:
                    resample_epoch_ieeg_bipolar = epoch_ieeg_bipolar
                else:
                    resample_epoch_ieeg_bipolar = mne.filter.resample(
                        epoch_ieeg_bipolar, down=SAMPLING_RATE / target_sr
                    ).astype(np.float32)
                epochs_ieeg_bipolar.append(resample_epoch_ieeg_bipolar)

                new_row = df_records.loc[i].to_dict()
                new_row['channel_name_1'] = esm_channel_1
                new_row['channel_name_2'] = esm_channel_2
                new_row['esm_type'] = int(esm_type) if esm_type != '-' else -1
                new_row['epoch_index'] = epoch_index_
                new_row['epoch_index_total'] = epoch_index_total_
                new_row[f'stimulus_onset_sr{SAMPLING_RATE}'] = stimulus_onset_
                new_row[f'stimulus_onset_sr{target_sr}'] = round(
                    stimulus_onset_ / SAMPLING_RATE * target_sr)
                new_row['voice_indices'] = voice_indices[epoch_index_total_]
                new_row['voice_valid'] = voice_valid[epoch_index_total_]
                new_row['voice_start1000'] = voice_indices[epoch_index_total_] if voice_valid[epoch_index_total_] else 0
                new_row['voice_start500'] = voice_indices[epoch_index_total_] // 2 if voice_valid[
                    epoch_index_total_] else 0
                df_epochs_electrodes_bipolar.append(new_row)

    df_epochs_electrodes_bipolar = pd.DataFrame(df_epochs_electrodes_bipolar)
    epochs_ieeg_bipolar = np.stack(epochs_ieeg_bipolar)

    df_epochs_electrodes_bipolar.to_csv(os.path.join(dirpath_datasets, 'df_epochs_electrodes_bipolar.csv'))

    np.save(os.path.join(dirpath_datasets, f'epochs_ieeg_bipolar{target_sr}_float32.npy'), epochs_ieeg_bipolar)

    return df_epochs_electrodes_bipolar, epochs_ieeg_bipolar


def create_full_dataset(
        datasets_dir_name, raw_directory, target_sr, baseline_duration_seconds, stimulus_duration_seconds,
        vad_fsds, vad_ignore_voice, vad_ignore_silence, vad_left_count, vad_threshold, vad_window,
        project_root
):

    dirpath_subject_sheets = os.path.join(project_root, raw_directory, 'sheets', 'subjects_sheets')
    df_subjects = pd.read_excel(
        os.path.join(project_root, raw_directory, 'sheets', 'Spmap Database progress.xlsx'), sheet_name='summary'
    )
    fif_directory = os.path.join(project_root, raw_directory, 'fif')
    dirpath_datasets = os.path.join(project_root, datasets_dir_name)
    os.makedirs(dirpath_datasets, exist_ok=True)

    df_records, dfs_active = build_df_records_and_active(
        df_subjects=df_subjects, dirpath_subject_sheets=dirpath_subject_sheets, fif_directory=fif_directory
    )
    epochs_sound4096 = get_epochs_sound(
        df_records=df_records, baseline_duration_seconds=baseline_duration_seconds,
        stimulus_duration_seconds=stimulus_duration_seconds, fif_directory=fif_directory
    )

    epochs_sound = mne.filter.resample(epochs_sound4096, down=SAMPLING_RATE / target_sr)

    np.save(os.path.join(dirpath_datasets, f'epochs_sound{target_sr}_float32.npy'), epochs_sound.astype(np.float32))

    vad = SimpleVAD(target_sr, fsds=vad_fsds, ignore_voice=vad_ignore_voice, ignore_silence=vad_ignore_silence)
    voice_detected40, voice_indices, voice_valid = vad.detect_multiple(
        epochs_sound=epochs_sound, window=vad_window, baseline_length=baseline_duration_seconds,
        left_count=vad_left_count, threshold=vad_threshold, dirpath_datasets=dirpath_datasets
    )

    df_epochs_electrodes = build_df_epochs_electrodes(
        df_records=df_records, dfs_active=dfs_active, voice_indices=voice_indices, voice_valid=voice_valid,
        target_sr=target_sr, baseline_duration_seconds=baseline_duration_seconds,
        stimulus_duration_seconds=stimulus_duration_seconds, dirpath_datasets=dirpath_datasets,
        fif_directory=fif_directory
    )
    df_records.to_csv(os.path.join(dirpath_datasets, 'df_records.csv'))

    get_ieeg_epochs(
        df_records=df_records, df_epochs_electrodes=df_epochs_electrodes, target_sr=target_sr,
        baseline_duration_seconds=baseline_duration_seconds, stimulus_duration_seconds=stimulus_duration_seconds,
        dirpath_datasets=dirpath_datasets, fif_directory=fif_directory
    )

    voice_indices[np.isnan(voice_indices)] = -1
    voice_indices = voice_indices.astype(int)

    get_bipolar_data(
        df_records=df_records, dfs_active=dfs_active, voice_indices=voice_indices, voice_valid=voice_valid,
        target_sr=target_sr, baseline_duration_seconds=baseline_duration_seconds,
        stimulus_duration_seconds=stimulus_duration_seconds, dirpath_datasets=dirpath_datasets,
        fif_directory=fif_directory
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description='Creating a dataset')

    parser.add_argument(
        '--datasets-dir-name', type=str, help='A path to the folder (from the project root) to save dataset',
        default='datasets_preprocessed', required=False
    )
    parser.add_argument(
        '--raw-directory', type=str, help='A path to the folder (from the project root) with raw data',
        default='datasets_raw', required=False
    )
    parser.add_argument(
        '--target-sr', type=int, help='Sampling rate to use for prepared dataset', default=1000, required=False
    )
    parser.add_argument(
        '--baseline-duration-seconds', type=int, help='A baseline duration in seconds', default=1, required=False
    )
    parser.add_argument(
        '--stimulus-duration-seconds', type=int, help='A stimulus duration in seconds', default=4, required=False
    )

    parser.add_argument(
        '--vad-fsds', type=int, help='fsds argument for VAD', default=40, required=False
    )
    parser.add_argument(
        '--vad-ignore-voice', type=int, help='ignore_voice argument for VAD', default=1, required=False
    )
    parser.add_argument(
        '--vad-ignore-silence', type=int, help='ignore_silence argument for VAD', default=7, required=False
    )
    parser.add_argument(
        '--vad-left_count', type=int, help='left_count argument for VAD', default=3, required=False
    )
    parser.add_argument(
        '--vad-threshold', type=int, help='threshold argument for VAD', default=100, required=False
    )
    parser.add_argument(
        '--vad-window-low', type=float, help='Lower bound for VAD window', default=0.75, required=False
    )
    parser.add_argument(
        '--vad-window-high', type=float, help='Higher bound for VAD window', default=3, required=False
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    create_full_dataset(
        datasets_dir_name=args.datasets_dir_name,
        raw_directory=args.raw_directory,
        target_sr=args.target_sr,
        baseline_duration_seconds=args.baseline_duration_seconds,
        stimulus_duration_seconds=args.stimulus_duration_seconds,
        vad_fsds=args.vad_fsds,
        vad_ignore_voice=args.vad_ignore_voice,
        vad_ignore_silence=args.vad_ignore_silence,
        vad_left_count=args.vad_left_count,
        vad_threshold=args.vad_threshold,
        vad_window=[args.vad_window_low, args.vad_window_high],
        project_root=PROJECT_ROOT
    )
