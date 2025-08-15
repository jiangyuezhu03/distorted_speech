#!/bin/bash

MODELS_DIR="/work/tc068/tc068/jiangyue_zhu/.cache/ft/"
KEYWORD=${1}  # space-separated keywords

output_file="${KEYWORD}_finetuned_model_list.txt"

# List only folders in MODELS_DIR, filter with grep, save to file
ls -d "$MODELS_DIR"/*/ | grep -E "$(echo "$KEYWORD" | sed 's/ /|/g')" | xargs -n 1 basename > "$output_file"

echo "Extracted ft models, saved to $output_file"
