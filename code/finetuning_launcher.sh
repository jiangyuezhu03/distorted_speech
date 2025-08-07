#!/bin/bash
# SLURM job settings
#SBATCH --job-name=distortion_finetuning
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=05:30:00
#SBATCH --account=tc068-pool2
#SBATCH --output=/work/tc068/tc068/jiangyue_zhu/log/%x_%j.out
#SBATCH --error=/work/tc068/tc068/jiangyue_zhu/log/%x_%j.err

# Set up environment variables
export HF_HOME="/work/tc068/tc068/jiangyue_zhu/.cache/huggingface"
export XDG_CACHE_HOME="/work/tc068/tc068/jiangyue_zhu/jetbrains_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HUB_OFFLINE=1 # uncomment for wavlm
ENV="new_test_env"
source /work/tc068/tc068/jiangyue_zhu/test_venv/$ENV/bin/activate
echo "activated $ENV"
WHISPER_SCRIPT="ft-whspr-small.py"
WAV2VEC_SCRIPT="ft-wav2vec2-large-xlsr.py"
WAVLM_SCRIPT="ft-wavlm-large.py"
#SCRIPTS=($WHISPER_SCRIPT $WAV2VEC_SCRIPT $WAVLM_SCRIPT)
#SCRIPT=$WAV2VEC_SCRIPT
# change to take argument for script

# single run
MODEL=${1}
#DISTORTION_TYPE=${2}
if [[ "$MODEL" == "whisper" ]]; then
    SCRIPT=$WHISPER_SCRIPT
elif [[ "$MODEL" == "wav2vec" ]]; then
    SCRIPT=$WAV2VEC_SCRIPT
elif [[ "$MODEL" == "wavlm" ]]; then
    SCRIPT=$WAVLM_SCRIPT
fi
DISTORTION_TYPE=("reversed" "narrowband" "narrowband_mid_only_2_3" "sinewave") # skip fast for wavlm
# for unfinished
#DISTORTION_TYPE=("reversed" "narrowband" "narrowband_mid_only_2_3" "sinewave")
# sweep run

for DIST in "${DISTORTION_TYPE[@]}"; do
    echo "Running $SCRIPT in $ENV on distortion: $DIST"
    srun python $SCRIPT $DIST
done

