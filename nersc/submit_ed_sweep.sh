#!/bin/bash
# Phase 3: real ED h_z-sweep as a Slurm job array on the `shared` QOS.
# `shared` charges only for the cores/memory requested (not a whole 512 GB node),
# which is right for the small (~3 GB) single-process ED job. Each array task
# computes one h_z point and writes a JSON file to $PSCRATCH.
#
#   sbatch nersc/submit_ed_sweep.sh
#
# Sweep resolution = size of --array. Index i -> h_z = i/10.
#SBATCH --job-name=tc-ed-hz
#SBATCH --account=mXXXX
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-10
#SBATCH --output=%x-%A_%a.out
set -euo pipefail

module load conda
conda activate tc-ed          # built by setup_conda_cpu.sh

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

HX=0.3
HZ=$(python -c "print(round(${SLURM_ARRAY_TASK_ID}/10, 3))")
OUTDIR=$PSCRATCH/tc_ed/hx${HX}
mkdir -p "$OUTDIR"

REPO=$HOME/Approximate-Symmetries-TC   # <-- EDIT to where you cloned the repo
cd "$REPO/nersc"

echo "task ${SLURM_ARRAY_TASK_ID}: hx=$HX hz=$HZ -> $OUTDIR"
HX=$HX HZ=$HZ L=2 OUT="$OUTDIR/ed_L2_hx${HX}_hz${HZ}.json" \
  srun -n 1 -c "$SLURM_CPUS_PER_TASK" python run_ed.py
