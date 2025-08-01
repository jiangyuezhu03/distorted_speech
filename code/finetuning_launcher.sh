#!/bin/bash
# SLURM job settings
#SBATCH --job-name=distortion_baseline
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
ENV="new_test_env"
source /work/tc068/tc068/jiangyue_zhu/test_venv/$ENV/bin/activate
echo "activated $ENV"
WHISPER_SCRIPT="ft-whspr-small.py"
WAV2VEC_SCRIPT="ft-wav2vec2-large-xlsr.py"
SCRIPTS=($WHISPER_SCRIPT $WAV2VEC_SCRIPT)

# change to take argument for script
MODEL=${1}
DISTORTION_TYPE=${2} # narrowband, reversed , sinewave

if [[ "$MODEL" == "wav2vec" ]]; then
    SCRIPT=$WHISPER_SCRIPT
elif [[ "$MODEL" == "whisper" ]]; then
    SCRIPT=$WAV2VEC_SCRIPT
fi
    for DIST in "${DISTORTION_TYPE[@]}"; do
        echo "Running $SCRIPT in $ENV on distortion: $DIST"
        srun python $SCRIPT $DIST
    done
done