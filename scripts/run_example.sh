#!/bin/bash
# Example launcher script. Copy and edit per experiment.
#
# Edit the variables below, then run from the repo root:
#   bash scripts/run_example.sh

# --- output identifiers (used in filenames) ---
OUTINDEX=3
JOBID="hx020_hz020"

# --- physics ---
LX=5                # number of vertices per side; OBC -> N = 2*Lx*(Lx-1) qubits
BC=PBC              # OBC or PBC
HX=0.0              # X field
HY=0.0              # Y field (nonzero -> sign problem -> complex dtype)
HZ=0.0              # Z field
J=1.0               # toric code coupling

# --- training (paper: lr 7e-3, diag_shift 5e-5) ---
DT=7e-3             # TDVP timestep == effective imaginary-time learning rate
SIM_TIME=0.7        # total imaginary time => n_iter = SIM_TIME/DT
DIAG_SHIFT=5e-5     # QGT regulariser; raise if training unstable

# --- network (paper: NIB [1,2,4] k=3 C-sigmoid, IB [4,4,4] k=15 C-ELU) ---
# Note: invariant kernel size (kernel_size_inv) is auto-set to Lx-1 in
#   utils/config.py:158 — at LX=5 that caps it at 4, not 15.  To literally
#   match the paper IB kernel you'd need LX >= 16.
ARCH=Combo                       # Combo or RPP
CHANNELS_NONINV=1,2,4              # NIB: in,hidden,out
CHANNELS_INV=4,4,4               # IB: first must equal CHANNELS_NONINV's last
KERNEL_SIZE=3                    # NIB kernel size

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
