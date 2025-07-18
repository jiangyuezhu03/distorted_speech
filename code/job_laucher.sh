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
# Define combinations
#ENVIRONMENTS=("my_test_env" "espnet_new")
WHISPER_SCRIPT="/work/tc068/tc068/jiangyue_zhu/code/whspr-small_baseline.py"
OWSMCTC_SCRIPT="/work/tc068/tc068/jiangyue_zhu/code/owsm-ctc_baseline.py"
OWSM4_SCRIPT="/work/tc068/tc068/jiangyue_zhu/code/owsm4_baseline.py"
WAVLM_SCRIPT="/work/tc068/tc068/jiangyue_zhu/code/wavlm-base_baseline.py"
WAV2VEC_SCRIPT="/work/tc068/tc068/jiangyue_zhu/code/wav2vec_batch_baseline.py"
#SCRIPTS=("$WHISPER_SCRIPT" "$OWSMCTC_SCRIPT")
#DISTORTIONS=("clean" "fast" "reversed" "narrowband" "tone_vocoded" "noise_vocoded" "sinewave" "glimpsed" "sculpted")
#DISTORTIONS=("fast" "reversed" "narrowband" "tone_vocoded" "noise_vocoded" "sinewave" "glimpsed" "sculpted")
SCRIPTS=($OWSMCTC_SCRIPT)

# iterate multiple models
for SCRIPT in "${SCRIPTS[@]}"; do
    if [[ "$SCRIPT" == "$WHISPER_SCRIPT" ||  "$SCRIPT" == "$WAVLM_SCRIPT" || "$SCRIPT" == "$WAV2VEC_SCRIPT" ]]; then
        ENV="my_test_env"
    elif [[ "$SCRIPT" == "$OWSMCTC_SCRIPT" || "$SCRIPT" == "$OWSM4_SCRIPT" ]]; then
        ENV="espnet_new"
        export PYTHONPATH=/work/tc068/tc068/jiangyue_zhu/espnet:$PYTHONPATH
    else
      echo "unknown model"
    fi

    source /work/tc068/tc068/jiangyue_zhu/test_venv/$ENV/bin/activate
    echo "activated $ENV"
#    for DIST in "${DISTORTIONS[@]}"; do
#        echo "Running $SCRIPT in $ENV on distortion: $DIST"
#        srun python $SCRIPT $DIST
#    done
#  CONFIGS=("low_mid_1_3" "high_mid_1_3" "low_high_1_3" "mid_only_1_3" "mid_only_2_3" "mid_only_1.0")
  CONFIGS=("0.5" "1.5" "2.5")

  DISTORTION_TYPE="fast"  # or any you want to fix or pass
  for CONDITION in ${CONFIGS[@]};do
    echo "Running condition: $CONDITION"
    python $SCRIPT $DISTORTION_TYPE $CONDITION
  done
  deactivate
done
# example use: sbatch job_launcher.sh

