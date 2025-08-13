#!/bin/bash

export PYTHONPATH=./

python src/evaluate_models.py --augmentations "with_sound" --models "svc" --low-freq 75 --high-freq 175