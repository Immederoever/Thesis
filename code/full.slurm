#!/bin/bash
#SBATCH --job-name=OPT0
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=18:00:00
#SBATCH --output=logs/OPT0_%j.out
#SBATCH --error=logs/OPT0_%j.err

module load 2023
module load CUDA/12.1.1
source /sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh
conda activate lm_dev2

cd ~/text_only

python FINAL_OPTIMIZED0.py
python EVAL.py FINAL_OPTIMIZED0_checkpoints > logs/EVAL_OPT0_${SLURM_JOB_ID}.out
