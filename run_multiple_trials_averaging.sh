for i in $(seq 0.2 0.1 0.9)
do
    python evaluate_models.py --augmentations "with_sound" --plots-save-dir "results_trials_bootstraps" --perc-trials $i --n-bootstraps 100 --models "logreg" "svc" "xgboost" "adaboost" --n-workers 1
done
