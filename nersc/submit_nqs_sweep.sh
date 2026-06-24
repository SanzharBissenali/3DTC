#!/bin/bash
# Phase 4: NQS hyperparameter sweep as a Slurm GPU job array. Each array task
# runs one config from scripts/sweep_params.py on a single A100 and logs the
# delta = |E - E_exact|/|E_exact| figure of merit to wandb; all tasks share one
# wandb group so they plot/compare side by side.
#
#   # size the array to the grid first:
#   N=$(python scripts/sweep_params.py --count)
#   HZ_PRESET=mid sbatch --array=0-$((N-1)) nersc/submit_nqs_sweep.sh
#
# HZ_PRESET (env) picks which hardcoded ED point to target: hard | mid | easy.
#SBATCH --job-name=tc-nqs-sweep
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --array=0-35           # default 36-combo grid; override with `sbatch --array=...`
#SBATCH --output=%x-%A_%a.out
set -euo pipefail

module load conda
conda activate tc-nqs          # built by setup_conda_gpu.sh; run `wandb login` once

REPO=$HOME/Approximate-Symmetries-TC   # <-- EDIT to where you cloned the repo
cd "$REPO"

HZ_PRESET="${HZ_PRESET:-mid}"
PARAMS=$(python scripts/sweep_params.py "${SLURM_ARRAY_TASK_ID}")

# If Perlmutter compute nodes cannot reach wandb.ai for your project, switch to
# offline and `wandb sync $PSCRATCH/wandb/offline-*` from a login node afterward:
#   export WANDB_MODE=offline

echo "task ${SLURM_ARRAY_TASK_ID}: hz_preset=$HZ_PRESET  params=$PARAMS"
srun -n 1 python -m Three_TC.train \
  --L 2 --bc PBC --model bosonic --arch ToricCNN_full \
  --hx 0.2 --hz_preset "$HZ_PRESET" \
  --n_iter 250 --n_samples 16384 --n_chains 1024 --qgt dense \
  --out_dir "$PSCRATCH/tc_nqs/${HZ_PRESET}" \
  --wandb_group "${SLURM_JOB_NAME}-${HZ_PRESET}" \
  --name "cfg${SLURM_ARRAY_TASK_ID}" \
  $PARAMS
