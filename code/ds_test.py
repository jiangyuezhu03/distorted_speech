# from datasets import load_from_disk,load_dataset, Dataset
# from transformers import WhisperTokenizer, WhisperFeatureExtractor
# import nemo_text_processing
import os
# from nemo_text_processing.text_normalization.normalize import Normalizer
# normalizer = Normalizer(input_case='cased', lang='en')
# feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
# tokenizer= WhisperTokenizer.from_pretrained("openai/whisper-small", language="english", task="transcribe")

# input_str= "for 80 years i lived" # not normalizing
# labels = tokenizer(input_str).input_ids
# decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
# decoded_str = tokenizer.decode(labels, skip_special_tokens=True)
#
# print(f"Input:                 {input_str}")
# print(f"Decoded w/ special:    {decoded_with_special}")
# print(f"Decoded w/out special: {decoded_str}")
# print(f"Are equal:             {input_str == decoded_str}")
# original_distorted_ds = load_from_disk("/work/tc068/tc068/jiangyue_zhu/ted3test_distorted/clean")
# new_adjusted_ds = load_from_disk("/work/tc068/tc068/jiangyue_zhu/ted3test_distorted_adjusted/narrowband_adjusted/narrowband_high_mid_1_3")
# ted3train=load_dataset("LIUM/tedlium", "release3", split="train")
#
# import random
# random.seed(8)
# # 3. Randomly sample 5000 indices
# sample_indices = random.sample(range(len(ted3train)), 5000)
# subset = ted3train.select(sample_indices)
# print("selected")
# subset.save_to_disk("../ted3train_5000")
# print("saved")

distortion_type="narrowband"
condition="mid_only_1_3"
print("start")
model_name = "/work/tc068/tc068/jiangyue_zhu/.cache/ft/owsm-ctc4_narrowband_cer_5e-05"

model_basename = model_name.split('/')[-1]
print(model_basename)
# Split into components
parts = model_basename.split('_')

model_short = parts[0].replace('whisper', 'whspr')  # converts "whisper" to "whspr"

# Check if this is a fine-tuned model (path contains '/ft/')
is_fine_tuned = '/ft/' in model_name

# Get the fine-tuning details (last two parts if fine-tuned)
ft_details = '_'.join(parts[-2:]) if is_fine_tuned else ''

# Construct the model identifier for output path
if is_fine_tuned:
    model_identifier = f"ft-{model_short}_{ft_details}"
else:
    model_identifier = model_short

# Now use this in your output path
output_path = f"/work/tc068/tc068/jiangyue_zhu/res/cer_res/{model_identifier}_{distortion_type}_{condition}_results.json"
print(output_path)
# /work/tc068/tc068/jiangyue_zhu/res/cer_res/ft-whspr-small_cer_5e-05_narrowband_mid_only_1_3_results.json
# /work/tc068/tc068/jiangyue_zhu/res/cer_res/ft-owsm-ctc4_cer_5e-05_narrowband_mid_only_1_3_results.json