#!/bin/bash
# Force "." as decimal separator regardless of system locale — otherwise
# awk emits "0,0000" on a comma-locale shell and argparse rejects it.
export LC_NUMERIC=C

# Sweep h_z at fixed h_x with the working main.py pipeline.
# Based on scripts/run_example.sh — same architecture and training params,
# wrapped in a loop over h_z values.  Each point writes its own JSON to
# outputs/ keyed by (hx, hz), so you can collect them post-hoc into a curve.
#
# Run from the repo root:
#   bash scripts/sweep_hz.sh

# --- output identifiers ---
OUTINDEX=3

# --- physics ---
LX=5                # number of vertices per side
BC=PBC              # OBC or PBC
HX=0.3              # fixed X field for the sweep
HY=0.0
J=1.0

# --- h_z grid ---
N_POINTS=20         # number of h_z values to sweep
HZ_MIN=0.0
HZ_MAX=0.75

# --- training (paper: lr 7e-3, diag_shift 5e-5) ---
DT=7e-3
SIM_TIME=0.7        # n_iter = SIM_TIME/DT
DIAG_SHIFT=5e-5

# --- network (paper: NIB [1,2,4] k=3 C-sigmoid, IB [4,4,4] k=15 C-ELU) ---
ARCH=Combo
CHANNELS_NONINV=1,2,4
CHANNELS_INV=4,4,4
KERNEL_SIZE=3

# --- sampling ---
N_SAMPLES=1024
N_SAMPLES_FIN=1024
CHUNK_SIZE=1024

# vertex+local custom MCMC (recommended for Lx>=4); drop the flag below if not wanted.
USE_CUSTOM_SAMPLER="--use_custom_sampler"

mkdir -p outputs outputs/logs
cd outputs

# 3-digit zero-padded label for h*100 (e.g. 0.30 -> "030", 0.075 -> "008").
hlabel() { awk -v v="$1" 'BEGIN{printf "%03d", v*100 + 0.5}'; }
HX_LABEL=$(hlabel "$HX")

SWEEP_START=$SECONDS

for step in $(seq 1 "$N_POINTS"); do
    i=$((step - 1))
    HZ=$(awk -v i=$i -v n=$N_POINTS -v lo=$HZ_MIN -v hi=$HZ_MAX \
           'BEGIN{printf "%.4f", lo + (hi - lo) * i / (n - 1)}')
    HZ_LABEL=$(hlabel "$HZ")
    JOBID="hx${HX_LABEL}_hz${HZ_LABEL}"

    printf "[%2d/%d] hz=%s  JOBID=%s ... " "$step" "$N_POINTS" "$HZ" "$JOBID"
    POINT_START=$SECONDS

    ../.venv/bin/python ../main.py \
      --outindex "$OUTINDEX" --jobid "$JOBID" \
      --Lx "$LX" --bc "$BC" \
      --hx "$HX" --hy "$HY" --hz "$HZ" --J "$J" \
      --dt "$DT" --sim_time "$SIM_TIME" --diag_shift "$DIAG_SHIFT" \
      --architecture "$ARCH" \
      --channels_noninv "$CHANNELS_NONINV" \
      --channels_inv "$CHANNELS_INV" \
      --kernel_size "$KERNEL_SIZE" \
      --n_samples "$N_SAMPLES" \
      --n_samples_fin "$N_SAMPLES_FIN" \
      --chunk_size "$CHUNK_SIZE" \
      $USE_CUSTOM_SAMPLER \
      > "logs/${JOBID}.log" 2>&1
    rc=$?

    DT_POINT=$((SECONDS - POINT_START))
    if [ $rc -eq 0 ]; then
        printf "done (%ds)\n" "$DT_POINT"
    else
        printf "FAILED rc=%d after %ds — see outputs/logs/%s.log\n" "$rc" "$DT_POINT" "$JOBID"
    fi
done

SWEEP_DT=$((SECONDS - SWEEP_START))
printf "Sweep finished in %dm %ds\n" $((SWEEP_DT / 60)) $((SWEEP_DT % 60))

echo "Sweep complete.  Result JSONs:"
ls -1 G-equiv_${OUTINDEX}_hx${HX_LABEL}_hz*.json 2>/dev/null
echo "Per-point logs in: outputs/logs/"
