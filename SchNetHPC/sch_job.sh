#!/bin/bash
#SBATCH --job-name=schnet_chelpg
#SBATCH -p general-gpu
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Output logs (relative to where job is submitted)
#SBATCH -o logs/schnet_train.log
#SBATCH -e logs/schnet_train.err

# ---------------- USER CONFIG ----------------
ENV_NAME="gnn_env"
SCRIPT_PATH="train_schnet.py" #change to eval_schnet.py when doing eval 
LOG_DIR="logs"


# --------------------------------------------

mkdir -p "$LOG_DIR"

source ~/miniconda3/bin/activate "$ENV_NAME" #user supply their miniconda env 

echo "Started: $(date)"
echo "Environment: $ENV_NAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python3 "$SCRIPT_PATH"

echo "Done: $(date)"
