#!/usr/bin/env bash
set -o pipefail

mkdir -p reviewer_logs

for attacker_seed in 41 42 43; do
  python experiments/gen_results/run_train_mia_attack_cvholdout_multigpu.py \
    --attack-data-dir reviewer_runs \
    --out "reviewer_results/learned_mia_seed${attacker_seed}" \
    --launcher \
    --test-ratio 0.2 \
    --cv-folds 5 \
    --tune \
    --n-trials 30 \
    --max-epochs 200 \
    --patience 15 \
    --device cuda \
    --seed "$attacker_seed" \
    --gpus 0,1,2,3,4,5,6,7 \
    --jobs-per-gpu 1 \
    --cpu-threads 1 \
    --resume \
    2>&1 | tee "reviewer_logs/learned_mia_seed${attacker_seed}.log"
done
