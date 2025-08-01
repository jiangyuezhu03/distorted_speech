# Import section stays the same
# write complete lr, e.g 1e-05 not just 1
import sys, os
from datasets import load_from_disk
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import torch
import json
from my_batch_eval import map_audio_and_text, map_batch_to_preds
from jiwer import cer
import numpy as np

# Set base model path
base_model_path = "/work/tc068/tc068/jiangyue_zhu/.cache/huggingface/hub/models--jonatasgrosman--wav2vec2-large-xlsr-53-english/snapshots/569a6236e92bd5f7652a0420bfe9bb94c5664080"

model_type = sys.argv[1]  # "base" or "ft"

if model_type == "base":
    if len(sys.argv) < 3:
        raise ValueError("Usage: base <distortion_type> [condition]")
    distortion_type = sys.argv[2]
    condition = sys.argv[3] if len(sys.argv) > 3 else None
    model_name = base_model_path
    model_identifier = "wav2vec2-large-xlsr"

elif model_type == "ft":
    if len(sys.argv) < 4:
        raise ValueError("Usage: ft <trained_on_distortion> <lr> [condition]")
    trained_on_distortion = sys.argv[2]
    lr = sys.argv[3]
    condition = sys.argv[4] if len(sys.argv) > 4 else None
    model_name = f"/work/tc068/tc068/jiangyue_zhu/.cache/ft/wav2vec2-large-xlsr_{trained_on_distortion}_cer_{lr}"
    model_basename = model_name.split("/")[-1]
    parts = model_basename.split("_")
    ft_details = '_'.join(parts[1:])
    model_identifier="".join(parts[:1]) # wav2vec2-large-xlsr
    print(f"model_identifier: {model_identifier}")
    model_output_identifier= f"{model_identifier}_{ft_details}"
    print("output identifier ",model_output_identifier)

    if "_" not in trained_on_distortion:
        distortion_type = trained_on_distortion
    else:
        distortion_type=trained_on_distortion.split("_")[0]
    print(f"distortions {distortion_type}")
else:
    raise ValueError("First argument must be either 'base' or 'ft'")
# Dataset path
if condition:
    dataset_path = f"../ted3test_distorted_adjusted/{distortion_type}_adjusted/{distortion_type}_{condition}"
    print(f"distortion_type: {distortion_type}, condition: {condition}")
    # Avoid repeating distortion_type in output file if condition already includes it
    output_path = f"/work/tc068/tc068/jiangyue_zhu/cer_res_norm_capped/{model_output_identifier}_{distortion_type}_{condition}_results.json"
else:
    dataset_path = f"../ted3test_distorted/{distortion_type}"
    output_path = f"/work/tc068/tc068/jiangyue_zhu/cer_res_norm_capped/{model_output_identifier}_{distortion_type}_results.json"

print(f"model name: {model_name}")
print(f"datapath: {dataset_path}")
print(f"output: {output_path}")
# 3. Now define the mapping function — it will now refer to *initialized* globals
def predict_batch(batch):
    return map_batch_to_preds(batch, model, processor, device)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'using {device}')
processor=Wav2Vec2Processor.from_pretrained(model_name)
model=Wav2Vec2ForCTC.from_pretrained(model_name).to(device).eval()
subset = load_from_disk(dataset_path)
subset = subset.map(map_audio_and_text,load_from_cache_file=False)
print("mapping")

result = subset.map(
    predict_batch,
    batched=True,
    batch_size=8, # reduce if cuda memory issue
    remove_columns=["audio", "text", "waveform", "sampling_rate"]
)
print("generating results")

results = {
    "segments": [],
    "overall_cer": None
}
cer_list=[]
cer_list_capped=[]
for ref, hyp in zip(result["transcript"], result["predicted"]):
    cer_score = round(cer(ref, hyp),4)
    capped_cer_score=min(cer_score,1.0)
    cer_list.append(cer_score)
    cer_list_capped.append(capped_cer_score)
    results["segments"].append({
        "reference": ref,
        "prediction": hyp,
        "cer": capped_cer_score
    })

results["overall_cer"] = cer(result["transcript"], result["predicted"])
results["average_cer"] = np.mean(cer_list_capped)
results["median_cer"] = np.median(cer_list)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved results to {output_path}")
