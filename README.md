# Towards Stimulation-Free Automatic Electrocorticographic Speech Mapping in Neurosurgery Patients

This is an official repository for "Towards Stimulation-Free Automatic Electrocorticographic Speech Mapping in 
Neurosurgery Patients" paper.

## Repository structure

This repository follows the next structure:
```     
├── constants.py                 # A Python file with constants related to data or experiments
├── create_dataset.py            # A Pyhton script which prepares raw data for the experiments
├── evaluate_models.py           # A Python script which runs all Monopolar experiments and saves the results 
├── evaluate_models_bipolar.py   # A Python script which runs all Bipolar experiments and saves the results 
├── validation.py                # A Pyhton file with utility functions for calculating results
├── datasets_raw                 # Directory for the dataset files
├── results                      # Directory created by default for images and tables with results
├── README.md                    # README file
└── requirements.txt             # A file with requirements 
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

After data was prepared you can run the following script for monopolar experiments:

```
python evaluate_models.py
```

To run bipolar experiments use:

```
python evaluate_models_bipolar.py
```

Both scripts will generate the results into the ```resutls``` folder (dy default)

To adjust experiments parameters check the arguments of the relative scripts

## Citation

If you use our data or code please cite our paper:
