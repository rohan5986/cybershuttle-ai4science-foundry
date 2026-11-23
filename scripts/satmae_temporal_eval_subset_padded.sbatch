#!/bin/bash
#SBATCH --job-name=satmae-temporal-subset-padded
#SBATCH -A coc
#SBATCH -p coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/satmae_temporal_subset_padded_%j.out
#SBATCH --error=logs/satmae_temporal_subset_padded_%j.err

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck source=satmae_temporal_gpu_common.sh
source "$REPO_ROOT/scripts/satmae_temporal_gpu_common.sh" subset_padded

PAD_ROOT="$SATMAE_TEMPORAL_WORKDIR/data/padded"
OUT_DIR="$SATMAE_TEMPORAL_WORKDIR/results/subset500-padded"
mkdir -p "$OUT_DIR"

START=$(date +%s)
python "$REPO_ROOT/scripts/satmae_temporal_evaluate.py" \
  --csv "$PAD_ROOT/val_prepared.csv" \
  --eval-csv "$PAD_ROOT/val_subset500.csv" \
  --checkpoint "$SATMAE_TEMPORAL_WORKDIR/weights/finetune_fmow_temporal.pth" \
  --code-dir "$SATMAE_CODE_DIR" \
  --out-dir "$OUT_DIR" \
  --batch-size 8 \
  --num-workers 8 \
  --seed 0
END=$(date +%s)
echo "wall_seconds=$((END-START))" | tee "logs/subset_padded_${SLURM_JOB_ID}_walltime.txt"

sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,MaxVMSize,State,ExitCode \
  > "logs/subset_padded_${SLURM_JOB_ID}_sacct.txt" || true
