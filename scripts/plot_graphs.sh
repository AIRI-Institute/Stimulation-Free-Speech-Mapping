#!/bin/bash

export PYTHONPATH=./

python src/plot_freq_heatmaps.py
python src/plot_percentage_dependency.py
python src/plot_weights.py
