#!/bin/bash

# default to 1, or take the first arg
n_workers="${1:-1}"

export PYTHONPATH=./

for i in $(seq 0.2 0.1 0.9); do
  python src/evaluate_models.py \
    --augmentations "with_sound" \
    --plots-save-dir "results/results_trials_bootstraps" \
    --perc-trials "$i" \
    --n-bootstraps 100 \
    --models "logreg" "svc" "xgboost" "adaboost" \
    --n-workers "$n_workers"
done
