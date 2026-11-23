#!/bin/bash
#SBATCH --job-name=satmae-temporal-crop-padded
#SBATCH -A coc
#SBATCH -p coc-cpu
#SBATCH --qos=coc-ice
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/satmae_temporal_crop_padded_%j.out
#SBATCH --error=logs/satmae_temporal_crop_padded_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p logs
# shellcheck source=satmae_temporal_paths.sh
source "$REPO_ROOT/scripts/satmae_temporal_paths.sh"
# shellcheck source=satmae_temporal_conda.sh
source "$REPO_ROOT/scripts/satmae_temporal_conda.sh"
satmae_activate_env

PAD_ROOT="$SATMAE_TEMPORAL_WORKDIR/data/padded"
mkdir -p "$PAD_ROOT"

python "$REPO_ROOT/scripts/satmae_temporal_crop_fmow.py" \
  --val-root "$SATMAE_TEMPORAL_WORKDIR/data/fmow/val" \
  --out-root "$PAD_ROOT/fmow/val" \
  --pad-context \
  --resize 224 \
  --overwrite \
  --workers "${SLURM_CPUS_PER_TASK:-8}"

python "$REPO_ROOT/scripts/satmae_temporal_prepare_csv.py" \
  --csv "$SATMAE_TEMPORAL_WORKDIR/data/val_62classes.csv" \
  --data-root "$PAD_ROOT" \
  --out "$PAD_ROOT/val_prepared.csv" \
  --subset-out "$PAD_ROOT/val_subset500.csv" \
  --subset-size 500 \
  --seed 0
