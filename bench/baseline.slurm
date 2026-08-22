#!/bin/bash
# Runs bench/baseline on one GPU. Submit from the repository root:
#   sbatch bench/baseline.slurm                       # A100
#   sbatch -p h200_normal_q -q tc_h200_normal_short bench/baseline.slurm
#SBATCH --job-name=cuein-baseline
#SBATCH --account=c36newriver
#SBATCH --partition=a100_normal_q
#SBATCH --qos=tc_a100_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=baseline-%j.out

set -euo pipefail
module load CUDA/12.6.0
module load NVHPC/25.1-CUDA-12.6.0

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo
./baseline
