#!/bin/bash
# One-time build of the CPU exact-diagonalisation environment on Perlmutter.
# Run this on a LOGIN node (not a compute node). Takes a few minutes.
#
#   bash nersc/setup_conda_cpu.sh
#
# Installs into project software space if PROJ is set (recommended by NERSC for
# faster import on compute nodes); otherwise into the default conda location.
set -euo pipefail

ENV_NAME=tc-ed
PROJ="${PROJ:-}"   # e.g. export PROJ=mXXXX  -> env lives in /global/common/software/$PROJ/conda

module load conda

if [[ -n "$PROJ" ]]; then
  TARGET="/global/common/software/$PROJ/conda/$ENV_NAME"
  echo "Creating env at $TARGET"
  conda create --yes --prefix "$TARGET" python=3.12
  conda activate "$TARGET"
else
  echo "Creating named env '$ENV_NAME' (default location)"
  conda create --yes --name "$ENV_NAME" python=3.12
  conda activate "$ENV_NAME"
fi

# ED stack only — no JAX/NetKet needed for scipy/numba exact diagonalisation.
pip install --no-cache-dir \
  "numpy==2.1.3" "scipy==1.15.2" numba tqdm

python - <<'PY'
import numpy, scipy, numba, scipy.sparse.linalg as spla
print("numpy", numpy.__version__, "scipy", scipy.__version__, "numba", numba.__version__)
print("eigsh importable:", spla.eigsh is not None)
PY
echo "Done. Activate later with:  module load conda && conda activate $ENV_NAME"
