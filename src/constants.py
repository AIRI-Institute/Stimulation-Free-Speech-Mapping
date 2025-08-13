import pathlib
import os


ESM_NEGATIVE = (0, )
ESM_POSITIVE = (1, 2)
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()
DATA_RAW = 'data/raw'
DATA_PREPROCESSED = 'data/preprocessed'
RESULTS = 'results'
MAIN_RESULTS = os.path.join(RESULTS, 'main_results')
TRIALS_RESULTS = os.path.join(RESULTS, 'results_trials_bootstraps')
VISUALIZATIONS = 'visualizations'
SAMPLING_RATE = 4096

SUBJECTS = [
    '211016', '211117', '220404', '220405', '220801', '220829', '220905', '221009', '221128', '230323', '230426',
    '230911', '231004', '231011'
]
