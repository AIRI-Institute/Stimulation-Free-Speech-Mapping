# Towards Stimulation-Free Automatic Electrocorticographic Speech Mapping in Neurosurgery Patients

This is an official repository for the "Towards Stimulation-Free Automatic Electrocorticographic Speech Mapping in 
Neurosurgery Patients" paper.

## Repository structure

This repository follows the structure below:

```
├── src/                                # All main Python code/scripts
│   ├── __init__.py                     # Empty file to make src a Python package
│   ├── constants.py                    # Constants related to data or experiments (centralizes all paths)
│   ├── create_dataset.py               # Prepares raw data for the experiments
│   ├── evaluate_models.py              # Runs all Monopolar experiments and saves the results 
│   ├── evaluate_models_bipolar.py      # Runs all Bipolar experiments and saves the results 
│   ├── plot_freq_heatmaps.py           # Plots heatmaps of metrics based on a filtering band during preprocessing
│   ├── plot_percentage_dependency.py   # Plots graphs of dependency of metrics on a percentage of trials
│   ├── plot_weights.py                 # Plots weights and patterns for select models and their correlations 
│   └── validation.py                   # Utility functions for calculating results 
│
├── scripts/                            # Bash scripts for running experiments and plotting
│   ├── plot_graphs.sh                  # Plots all graphs for results visualization
│   ├── run_main_experiments.sh         # Runs main experiments for the paper
│   ├── run_multiple_frequencies.sh     # Runs experiments for different frequency bands
│   ├── run_multiple_trials_averaging.sh# Runs experiments with bootstraps and different percentages of trials 
│   └── setup_environment.sh            # Sets up the environment (e.g., conda, pip)
│
├── data/                               # Data folders (not included in repo, see README for download instructions)
│   ├── raw/                            # Directory for the raw dataset files
│   └── preprocessed/                   # Directory with preprocessed dataset files, created by create_dataset.py
│
├── results/                            # Directory for images (ROC and PR curves) and tables with results  
│   ├── main_results/                   # Directory for results of main experiments  
│   ├── results_trials_bootstraps/      # Directory for results of experiments with different percentages of trials  
├── visualizations/                     # Directory for graphs of results plotted with plotting scripts  
│
├── LICENSE                             # License file
├── README.md                           # This file
├── requirements.txt                    # Python requirements 
└── .gitignore                          # Specifies files/folders to ignore in git
```

## Dataset

For this project we used our dataset which you can download [here](https://osf.io/xuegw/)  

Download data and put it into the ```data/raw``` folder in the repository.

If you utilize this data in your study please credit our paper.

## Environment setup

To set up the conda environment you can run 

```
bash scripts/setup_environment.sh
``` 

or use the following code:

```
conda create -n Speech-Mapping python=3.12.7
conda activate Speech-Mapping
pip install -r requirements.txt
```

## Preparing dataset

Before you can run our experiments you have to prepare raw dataset. To do this run:

```
python src/create_dataset.py
```

To adjust some parameters (e.g. directory paths) check the arguments of the script

## Running experiments

### Main results

After data was prepared you can run the following script for monopolar experiments:

```
python src/evaluate_models.py
```

To run bipolar experiments use:

```
python src/evaluate_models_bipolar.py
```

Both scripts will generate the results into the ```resutls/main_results``` folder (dy default)

This will run the scripts with the default parameters to obtain our main results from the paper for 100-150 Hz frequency band.
To adjust experiments parameters check the arguments of the relative scripts

Alternatively, you can run prepared bash script:

```
bash scripts/run_main_experiments.sh
```

### Exploring frequency bands

To run experiments and obtain metrics for different frequency bands (like shown on a heatmap in our paper), use the following setup:

```
python src/evaluate_models.py --augmentations "with_sound" --models "svc" --low-freq 75 --high-freq 175
```

This will run many experiments for different frequency bands. To use multiple processes and speed-up computations use argument ```--n-workers``` (by default is 1).

Results will be available in the ```results/main_results``` folder (by default)

Alternatively, you can run prepared bash script:

```
bash scripts/run_multiple_frequencies.sh
```

### Exploring different amounts of trials

To verify the difference in performance when a new left out subject has different amount of trials for averaging we perform experiments where we randomly sample a chosen percentage of trials. We do these 100 times for each percentage. To run these experiments as described in the paper run the following script:

```
bash scripts/run_multiple_trials_averaging.sh {n_workers}
```

This will run many experiments for different percentages and also uses bootstrap. To use multiple processes and speed-up computations pass the amount of workers to the bash script (by default there is 1 worker).

Results will be available in the ```results/results_trials_bootstraps``` folder (by default)

## Visualizing the results

Note: all of the plotting scripts assume default directories and experiment parameters as used above. 
If you changed any parameters during computations make sure that scripts pull results from correct places.

To get all visualizations you can run:

```
bash scripts/plot_graphs.sh
```

### Main results

Tables and ROC curves with main results are saved into experiment results directory (```results``` by default)

### Exploring frequency bands

To plot heatmaps with quality metrics vs the frequency band run the following script:

```python src/plot_freq_heatmaps.py```

Graphs will appear in the ```visualizations/heatmaps``` directory (by default)

### Exploring different amounts of trials

To plot graphs with quality metrics vs the percentage of trials for averaging run the following script:

```python src/plot_percentage_dependency.py```

Graphs will appear in the ```visualizations/perc-test-trials-bootstraps``` directory (by default)

### Exploring patterns

To plot graphs with patterns and their correlations run the following script:

```python src/plot_weights.py```

Graphs will appear in the ```visualizations/feature_interpretation``` directory (by default)

Note: such patterns could only be constructed for SVC and the Logistic Regression due to the nature of the models and patterns definition. 

## Citation

If you use our data or code please cite our paper:
