import numpy as np
from standardize_text import standardize_reference_text, clean_punctuations_transcript_whspr
import torch
import torchaudio
import json
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from my_batch_eval import map_audio_and_text, map_batch_to_preds_whisper
from datasets import load_dataset,load_from_disk
from tqdm import tqdm
import sys

distortion_type=sys.argv[1]
condition = sys.argv[2]
# model_name = "openai/whisper-small"
model_name="/work/tc068/tc068/jiangyue_zhu/.cache/ft/whisper-small_narrowband_cer_5e-05"
model_basename = model_name.split('/')[-1]
# Split into components
parts = model_basename.split('_')
model_short = parts[0].replace('whisper', 'whspr')  # converts "whisper" to "whspr"

# Check if this is a fine-tuned model (path contains '/ft/')
is_fine_tuned = '/ft/' in model_name
# Get the fine-tuning details (last two parts if fine-tuned)
ft_details = '_'.join(parts[-2:]) if is_fine_tuned else ''

if is_fine_tuned:
    model_identifier = f"ft-{model_short}_{ft_details}"
else:
    model_identifier = model_short
# output_path = f"/work/tc068/tc068/jiangyue_zhu/res/whspr-small_{distortion_type}_results.json"
# output_path = f"/work/tc068/tc068/jiangyue_zhu/res/cer_res/whspr-small_{distortion_type}_{condition}_results.json"
output_path = f"/work/tc068/tc068/jiangyue_zhu/res/cer_res/{model_identifier}_{distortion_type}_{condition}_results.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'using {device}')
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device).eval()
# make sure uses Eng
forced_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
model.config.forced_decoder_ids = forced_ids

results = {"segments": [], "overall_wer": 0.0}
predictions = []
references = []

# subset = load_dataset("LIUM/tedlium", "release3", split="test",trust_remote_code = True)
# subset = load_from_disk(f"/work/tc068/tc068/jiangyue_zhu/ted3test_distorted/{distortion_type}")
# the adjusted dataset
dataset_path = f"../ted3test_distorted_adjusted/{distortion_type}_adjusted/{distortion_type}_{condition}"
subset = load_from_disk(dataset_path)
subset = subset.map(map_audio_and_text)
print("mapping")

def predict_batch(batch):
    return map_batch_to_preds_whisper(batch, model, processor, device,forced_ids)

result = subset.map(
        predict_batch,
        batched=True,
        batch_size=16,
        remove_columns=["audio", "text", "waveform", "sampling_rate"]
    )
print("generating results")

results = {
    "segments": [],
    "overall_cer": None
}

for ref, hyp in zip(result["transcript"], result["predicted"]):
    results["segments"].append({
        "reference": ref,
        "prediction": clean_punctuations_transcript_whspr(hyp),
        "cer": None
    })


with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved results to {output_path}")
