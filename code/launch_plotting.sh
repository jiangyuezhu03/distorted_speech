#!/bin/bash
# SLURM job settings
#SBATCH --job-name=plotting
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --account=tc068-pool2
#SBATCH --output=/work/tc068/tc068/jiangyue_zhu/log/%x_%j.out
#SBATCH --error=/work/tc068/tc068/jiangyue_zhu/log/%x_%j.err

# Set up environment variables
export HF_HOME="/work/tc068/tc068/jiangyue_zhu/.cache/huggingface"
export XDG_CACHE_HOME="/work/tc068/tc068/jiangyue_zhu/jetbrains_cache"
export HF_HUB_OFFLINE=1 # uncomment for wavlm
# Define combinations
ENV="new_test_env"
source /work/tc068/tc068/jiangyue_zhu/test_venv/$ENV/bin/activate
echo "activated $ENV"
srun python layer_comparison.py "narrowband"



# example use: sbatch job_launcher.sh
