#!/bin/bash
cd /work/tc068/tc068/jiangyue_zhu/code
# SLURM job settings
#SBATCH --job-name=distortion_baseline
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=tc068-pool2
#SBATCH --output=/work/tc068/tc068/jiangyue_zhu/log/%x_%j.out
#SBATCH --error=/work/tc068/tc068/jiangyue_zhu/log/%x_%j.err

# Set up environment variables
export HF_HOME="/work/tc068/tc068/jiangyue_zhu/.cache/huggingface"
export XDG_CACHE_HOME="/work/tc068/tc068/jiangyue_zhu/jetbrains_cache"
export HF_HUB_OFFLINE=1
# Define combinations
ENVIRONMENTS=("my_test_env" "espnet_new")
SCRIPTS=("/work/tc068/tc068/jiangyue_zhu/code/whspr-small_baseline.py" "/work/tc068/tc068/jiangyue_zhu/code/owsm-ctc_baseline.py")

DISTORTIONS=("fast" "reversed" "tone_vocoded" "noise_vocoded" "sinewave" "glimpsed" "sculpted")

for SCRIPT in "${SCRIPTS[@]}"; do
    if [[ "$SCRIPT" == *whspr* ]]; then
        ENV="my_test_env"
    else
        ENV="espnet_new"
    fi

    source /work/tc068/tc068/jiangyue_zhu/test_venv/$ENV/bin/activate

    for DIST in "${DISTORTIONS[@]}"; do
        echo "Running $SCRIPT in $ENV on distortion: $DIST"
        srun python $SCRIPT $DIST
    done

    deactivate
done

