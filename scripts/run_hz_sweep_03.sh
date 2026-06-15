#!/bin/bash
# Example launcher script. Copy and edit per experiment.
#
# Edit the variables below, then run from the repo root:
#   bash scripts/run_example.sh

# --- output identifiers (used in filenames) ---
OUTINDEX=5
JOBID="L4_hz020"

# --- physics ---
LX=4                # number of vertices per side; OBC -> N = 2*Lx*(Lx-1) qubits
BC=OBC              # OBC or PBC
HX=0.0              # X field
HY=0.0              # Y field (nonzero -> sign problem -> complex dtype)
HZ=0.2              # Z field
J=1.0               # toric code coupling

# --- training ---
DT=0.02             # TDVP timestep
SIM_TIME=2.0        # total imaginary time => n_iter = SIM_TIME/DT
DIAG_SHIFT=1e-4     # QGT regulariser; raise if training unstable

# --- network ---
ARCH=Combo                       # Combo or RPP
CHANNELS_NONINV=1,4              # comma-sep: in,hidden1,...,out
CHANNELS_INV=4,4,1               # first must equal CHANNELS_NONINV's last
KERNEL_SIZE=1

# --- sampling ---
N_SAMPLES=1024
N_SAMPLES_FIN=1024
CHUNK_SIZE=1024
# Add --use_custom_sampler below for vertex-update MCMC (recommended for Lx>=4)

mkdir -p outputs
cd outputs

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
  --chunk_size "$CHUNK_SIZE"
