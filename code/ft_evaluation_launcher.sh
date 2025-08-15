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
WHISPER_EVAL="/work/tc068/tc068/jiangyue_zhu/code/whspr-small-ft_evaluation.py"
WAV2VEC_EVAL="/work/tc068/tc068/jiangyue_zhu/code/wav2vec-ft_evaluation.py"

ENV="new_test_env"
source /work/tc068/tc068/jiangyue_zhu/test_venv/$ENV/bin/activate
echo "activated $ENV"

MODEL_TYPE=${1}
#FT_MODELS= the list read from ${MODEL_TYPE}_finetuend_model_list.txt
DISTORTION_TYPE=${2}
mapfile -t FT_MODELS < "/work/tc068/tc068/jiangyue_zhu/code/${MODEL_TYPE}_finetuned_model_list.txt"

echo "Loaded ${#FT_MODELS[@]} models from ${MODEL_TYPE}_finetuned_model_list.txt"

if [[ "$MODEL_TYPE" == "whisper" ]]; then
    SCRIPT="$WHISPER_EVAL"
elif [[ "$MODEL_TYPE" == "wav2vec" ]]; then
    SCRIPT="$WAV2VEC_EVAL"
else
    echo "Unknown model type: $MODEL_TYPE"
    exit 1
fi


if [[ "$DISTORTION_TYPE" == "narrowband" ]]; then
  # add the new conditions
    CONFIGS=("low_mid_1_3" "high_mid_1_3" "low_high_1_3" "mid_only_1_3" "mid_only_2_3" "mid_only_1.0" "all_bands_1_3")
#        CONFIGS=("all_bands_1_3") #
# no new condition for FAST
elif [[ "$DISTORTION_TYPE" == "fast" ]]; then
    CONFIGS=("0.5" "1.5" "2.5")
# add new window sizes for reversed, models except for whisper haven't run on adjusted reversed
elif [[ "$DISTORTION_TYPE" == "reversed" ]]; then
#        CONFIGS=("20ms" "31ms" "62ms")
    CONFIGS=("20ms" "31ms" "62ms" "40ms" "50ms" "80ms")
else
    CONFIGS=()  # No condition configs needed
fi

for MODEL in "${FT_MODELS[@]}"; do

  # script.py model distortion conidtion
  if [ ${#CONFIGS[@]} -eq 0 ]; then
      # Run with no condition argument
      echo "Running distortion: $DISTORTION_TYPE with no condition"
      python $SCRIPT $MODEL $DISTORTION_TYPE
  else
      for CONDITION in "${CONFIGS[@]}"; do
          echo "Running condition: $CONDITION"
          python $SCRIPT $MODEL $DISTORTION_TYPE $CONDITION
      done
  fi
done
