#!/bin/bash
#SBATCH --job-name=prithvi-probe
#SBATCH -A coc
#SBATCH -p coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/prithvi_probe_%j.out
#SBATCH --error=logs/prithvi_probe_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p logs
# shellcheck source=satmae_temporal_conda.sh
source "$REPO_ROOT/scripts/satmae_temporal_conda.sh"
satmae_source_conda

export PRITHVI_ENV="${PRITHVI_ENV:-$HOME/scratch/envs/prithvi-eo}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf-cache}"
export SATMAE_TEMPORAL_WORKDIR="${SATMAE_TEMPORAL_WORKDIR:-$HOME/scratch/satmae-temporal}"
set +u
conda activate "$PRITHVI_ENV"
set -u

RGB=$(python - <<'PY'
from pathlib import Path
import csv
root = Path.home() / "scratch/satmae-temporal/data"
path = root / "val_subset500.csv"
hit = ""
if path.is_file():
    with path.open() as f:
        rows = csv.reader(f)
        next(rows, None)
        for row in rows:
            if len(row) < 2:
                continue
            cand = root / row[1]
            if cand.is_file():
                hit = str(cand)
                break
print(hit)
PY
)

OUT="$SATMAE_TEMPORAL_WORKDIR/results/prithvi-probe/evidence.json"
mkdir -p "$(dirname "$OUT")"
echo "rgb=$RGB device=cuda"
python "$REPO_ROOT/scripts/prithvi_probe.py" --rgb "$RGB" --out "$OUT" --device cuda
