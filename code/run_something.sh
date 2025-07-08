#!/bin/bash

ENV_NAME=$1       # e.g., my_test_env
SCRIPT_NAME=$2    # e.g., ds_test.py
shift 2
# Activate environment
source /work/tc068/tc068/jiangyue_zhu/test_venv/$ENV_NAME/bin/activate
echo "activated $ENV_NAME"
# Redirect Hugging Face and JetBrains cache to work disk
export HF_HOME="/work/tc068/tc068/jiangyue_zhu/.cache/huggingface"
export XDG_CACHE_HOME="/work/tc068/tc068/jiangyue_zhu/jetbrains_cache"
export HF_HUB_OFFLINE=1
# Run the specified Python script
python $SCRIPT_NAME "$@"

deactivate