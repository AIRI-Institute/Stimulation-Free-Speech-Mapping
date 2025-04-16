
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

## Environment setup

To set up the conda environment run the following code:

```
conda create -n Speech-Mapping python=3.12.7
conda activate Speech-Mapping
pip install -r requirements.txt
```

