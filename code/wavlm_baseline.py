import numpy as np
import torch
import torchaudio
import json
from standardize_text import standardize_reference_text
from transformers import AutoProcessor, WavLMForCTC
from datasets import load_from_disk
from my_batch_eval import map_audio_and_text, map_batch_to_preds
import sys
from jiwer import cer
from tqdm import tqdm

distortion_type=sys.argv[1] # must match folder names
condition = sys.argv[2] if len(sys.argv)>2 else None
if condition: # with normalizer, now save to processed folder directly
    output_path = f"/work/tc068/tc068/jiangyue_zhu/cer_res_norm_capped/wavlm-large_{distortion_type}_{condition}_results_cer.json"
    dataset_path= f"../ted3test_distorted_adjusted/{distortion_type}_adjusted/{distortion_type}_{condition}"
else: # not using "adjusted"
    output_path = f"/work/tc068/tc068/jiangyue_zhu/cer_res_norm_capped/wavlm-large_{distortion_type}_results_cer.json"
    dataset_path= f"../ted3test_distorted/{distortion_type}"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")

# Load model and processor: processor throws an error
# model_name = "patrickvonplaten/wavlm-libri-clean-100h-large"
model_name = "../.cache/huggingface/hub/models--patrickvonplaten--wavlm-libri-clean-100h-large/snapshots/e70e3a062ec399c46008ee55d1fb52c7ba338d5c"#
processor = AutoProcessor.from_pretrained(model_name,local_files_only=True)#local_files_only = True,
model = WavLMForCTC.from_pretrained(model_name,  use_safetensors=True, local_files_only=True).to(device).eval()

subset = load_from_disk(dataset_path)
print("loaded dataset")
subset = subset.map(map_audio_and_text)

results = {"segments": [], "overall_wer": None}
predictions = []
references = []

def predict_batch(batch):
    return map_batch_to_preds(batch, model, processor, device)

result = subset.map(
    predict_batch,
    batched=True,
    batch_size=16,
    remove_columns=["audio", "text", "waveform", "sampling_rate"],
    num_proc=1,
    load_from_cache_file=False
)
print("generating results")
cer_list = []
cer_list_capped = []
results = {
    "segments": [],
    "overall_cer": None
}

for ref, hyp in zip(result["transcript"], result["predicted"]):
    cer_score = cer(ref, hyp)
    cer_list.append(cer_score)
    cer_list_capped.append(cer_score)
    results["segments"].append({
        "reference": ref,
        "prediction": hyp,
        "cer": min(cer_score, 1.0)
    })

results["overall_cer"] = cer(result["transcript"], result["predicted"])
results["avg_cer"] = np.mean(cer_list_capped)
results["median_cer"] = np.median(cer_list)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved results to {output_path}")
