#!/bin/bash
# Separate env from SatMAE (torch 1.12). Prithvi-EO-2.0 needs a current stack.
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
# shellcheck source=satmae_temporal_conda.sh
source "$REPO_ROOT/scripts/satmae_temporal_conda.sh"
satmae_source_conda

export PRITHVI_ENV="${PRITHVI_ENV:-$HOME/scratch/envs/prithvi-eo}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf-cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$HOME/scratch/conda-pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/scratch/pip-cache}"
mkdir -p "$HF_HOME" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$(dirname "$PRITHVI_ENV")"

if [[ ! -x "$PRITHVI_ENV/bin/python" ]]; then
  conda create -y -p "$PRITHVI_ENV" python=3.11 pip
fi
set +u
conda activate "$PRITHVI_ENV"
set -u
pip install --upgrade pip
# Official IBM/NASA load path: TerraTorch BACKBONE_REGISTRY
pip install "torch>=2.2" torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "terratorch>=1.1" pillow numpy

python - <<'PY'
import torch
from terratorch.registry import BACKBONE_REGISTRY
print("python_ok torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("registry has prithvi", any("prithvi" in n.lower() for n in dir(BACKBONE_REGISTRY)))
PY
echo "prithvi env: $PRITHVI_ENV"
