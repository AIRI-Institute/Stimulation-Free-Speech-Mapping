# Towards Stimulation-Free Automatic Electrocorticographic Speech Mapping in Neurosurgery Patients

This is an official repository for the "Towards Stimulation-Free Automatic Electrocorticographic Speech Mapping in 
Neurosurgery Patients" paper.

## Repository structure

This repository follows the next structure:
```     
├── constants.py                     # A Python file with constants related to data or experiments
├── create_dataset.py                # A Pyhton script which prepares raw data for the experiments
├── evaluate_models.py               # A Python script which runs all Monopolar experiments and saves the results 
├── evaluate_models_bipolar.py       # A Python script which runs all Bipolar experiments and saves the results 
├── plot_freq_heatmaps.py            # A Pyhton script which plots heatmaps of metrics based on a filtering band during preprocessing
├── plot_percentage_dependency.py    # A Pyhton script which plots graphs of dependency of metrics on a percentage of trials
├── plot_weights.py                  # A Pyhton script which plots weights and patterns for select models and their correlations 
├── validation.py                    # A Pyhton file with utility functions for calculating results 
├── run_multiple_trials_averaging.sh # A bash script for running experiments with bootstraps and different percentages of trials 
├── datasets_raw                     # Directory for the dataset files
├── datasets_preprocessed            # Directory with preprocessed dataset files that is created by create_dataset.py
├── results                          # Directory created by default for images (ROC and PR curves) and tables with results  
├── results_trials_bootstraps        # Directory created by default for images (ROC and PR curves) and tables with results for experiments with different percentage of trials  
├── visualization                    # Directory created by default for graphs of results plotted with plot_freq_heatmaps.py, plot_percentage_dependency.py and plot_weights.py  
├── LICENCE                          # LICENCE file
├── README.md                        # README file
└── requirements.txt                 # A file with requirements 
```

## Dataset

For this project we used our dataset which you can download [here](https://osf.io/xuegw/)  

Download ```datasets_raw``` folder and put it into the root of the repository.

If you utilize this data in your study please credit our paper.

## Environment setup

To set up the conda environment run the following code:

```
conda create -n Speech-Mapping python=3.12.7
conda activate Speech-Mapping
pip install -r requirements.txt
```

## Preparing dataset

Before you can run our experiments you have to prepare raw dataset. To do this run:

```
python create_dataset.py
```

To adjust some parameters (e.g. directory paths) check the arguments of the script

## Running experiments

### Main results

After data was prepared you can run the following script for monopolar experiments:

```
python evaluate_models.py
```

To run bipolar experiments use:

```
python evaluate_models_bipolar.py
```

Both scripts will generate the results into the ```resutls``` folder (dy default)

This will run the scripts with the default parameters to obtain our main results from the paper for 100-150 Hz frequency band.
To adjust experiments parameters check the arguments of the relative scripts

### Exploring frequency bands

To run experiments and obtain metrics for different frequency bands (like shown on a heatmap in our paper), use the following setup:

```
python evaluate_models.py --augmentations "with_sound" --models "svc" --low-freq 75 --high-freq 175
```

This will run many experiments for different frequency bands. To use multiple processes and speed-up computations use argument ```--n-workers``` (by default is 1).

Results will be available in the ```results``` folder (by default)

### Exploring different amounts of trials

To verify the difference in performance when a new left out subject has different amount of trials for averaging we perform experiments where we randomly sample a chosen percentage of trials. We do these 100 times for each percentage. To run these experiments as described in the paper run the following script:

```
bash run_multiple_trials_averaging.sh
```

This will run many experiments for different percentages and also uses bootstrap. To use multiple processes and speed-up computations change the argument ```--n-workers``` in the bash script (by default is 1).

Results will be available in the ```results_trials_bootstraps``` folder (by default)

## Visualizing the results

Note: all of the plotting scripts assume default directories and experiment parameters as used above. 
If you changed any parameters during computations make sure that scripts pull results from correct places.

### Main results

Tables and ROC curves with main results are saved into experiment results directory (```results``` by default)

### Exploring frequency bands

To plot heatmaps with quality metrics vs the frequency band run the following script:

```python plot_freq_heatmaps.py```

Graphs will appear in the ```visualization/heatmaps``` directory (by default)

### Exploring different amounts of trials

To plot graphs with quality metrics vs the percentage of trials for averaging run the following script:

```python plot_percentage_dependency.py```

Graphs will appear in the ```visualization/perc-test-trials-bootstraps``` directory (by default)

### Exploring patterns

To plot graphs with patterns and their correlations run the following script:

```python plot_weights.py```

Graphs will appear in the ```visualization/feature_interpretation``` directory (by default)

Note: such patterns could only be constructed for SVC and the Logistic Regression due to the nature of the models and patterns definition. 

## Citation

If you use our data or code please cite our paper:
