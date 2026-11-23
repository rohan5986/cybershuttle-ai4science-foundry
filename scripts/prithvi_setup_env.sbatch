#!/bin/bash
#SBATCH --job-name=prithvi-env
#SBATCH -A coc
#SBATCH -p coc-cpu
#SBATCH --qos=coc-ice
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prithvi_setup_env_%j.out
#SBATCH --error=logs/prithvi_setup_env_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p logs
bash "$REPO_ROOT/scripts/prithvi_setup_env.sh"
