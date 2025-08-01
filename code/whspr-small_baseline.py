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

model_type = sys.argv[1]  # "base" or "ft"

if model_type == "base":
    if len(sys.argv) < 3:
        raise ValueError("Usage: base <distortion_type> [condition]")
    distortion_type = sys.argv[2]
    condition = sys.argv[3] if len(sys.argv) > 3 else None
    model_name = "openai/whisper-small"
    model_identifier = "whspr-small"

elif model_type == "ft":
    if len(sys.argv) < 4:
        raise ValueError("Usage: ft <enc|full> <trained_on_distortion> <lr> [condition]")
    training_scope = sys.argv[2]  # "enc" or "full"
    trained_on_distortion = sys.argv[3]
    lr = sys.argv[4]
    condition = sys.argv[5] if len(sys.argv) > 5 else None
    if training_scope == "enc":
        model_name = f"/work/tc068/tc068/jiangyue_zhu/.cache/ft/whisper-small_enc_{trained_on_distortion}_cer_{lr}"
        model_identifier = f"ft-whisper-small_enc_{trained_on_distortion}_cer_{lr}"
    elif training_scope == "full":
        # skip 'full' in the model path
        model_name = f"/work/tc068/tc068/jiangyue_zhu/.cache/ft/whisper-small_{trained_on_distortion}_cer_{lr}"
        model_identifier = f"ft-whisper-small_{trained_on_distortion}_cer_{lr}"
    else:
        raise ValueError("training_scope must be 'enc' or 'full'")
    # For now, assume lr is fixed
    model_name = f"/work/tc068/tc068/jiangyue_zhu/.cache/ft/whisper-small_{training_scope}_{trained_on_distortion}_cer_{lr}"
    model_basename = model_name.split("/")[-1]
    parts = model_basename.split("_")
    model_short = parts[0].replace("whisper", "whspr")
    ft_details = '_'.join(parts[1:])  # everything after 'whisper-small'
    print(f"details {ft_details}")
    model_output_identifier = f"ft-{model_short}_{ft_details}"
    print(f"output identifier {model_output_identifier}")

    # Evaluation distortion type is same as trained-on unless you change it
    if "_" not in trained_on_distortion:
        distortion_type = trained_on_distortion
    else:
        distortion_type=trained_on_distortion.split("_")[0]
    print(f"distortions {distortion_type}")

else:
    raise ValueError("First argument must be either 'base' or 'ft'")

# Final output path
if condition:
    dataset_path = f"../ted3test_distorted_adjusted/{distortion_type}_adjusted/{distortion_type}_{condition}"
    print(f"distortion_type: {distortion_type}, condition: {condition}")
    # Avoid repeating distortion_type in output file if condition already includes it
    if model_type == "ft":
        output_path = f"/work/tc068/tc068/jiangyue_zhu/res/cer_res/{model_output_identifier}_{condition}_results.json"
    else:
        output_path = f"/work/tc068/tc068/jiangyue_zhu/res/cer_res/{model_identifier}_{distortion_type}_{condition}_results.json"
else:
    dataset_path = f"../ted3test_distorted/{distortion_type}"
    output_path = f"/work/tc068/tc068/jiangyue_zhu/res/cer_res/{model_output_identifier}_{distortion_type}_results.json"

print(f"model name: {model_name}")
print(f"datapath: {dataset_path}")
print(f"output: {output_path}")
# import pdb;pdb.set_trace()
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
