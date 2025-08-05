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
model_name = "/work/tc068/tc068/jiangyue_zhu/.cache/huggingface/hub/models--jonatasgrosman--wav2vec2-large-xlsr-53-english/snapshots/569a6236e92bd5f7652a0420bfe9bb94c5664080"

distortion_type=sys.argv[1] # must match folder names
condition = sys.argv[2] if len(sys.argv)>2 else None
if condition: # with normalizer, now save to processed folder directly
    output_path = f"/work/tc068/tc068/jiangyue_zhu/cer_res_norm_capped/wav2vec2-large-xlsr_{distortion_type}_{condition}_results_cer.json"
    dataset_path= f"../ted3test_distorted_adjusted/{distortion_type}_adjusted/{distortion_type}_{condition}"
else: # not using "adjusted"
    output_path = f"/work/tc068/tc068/jiangyue_zhu/cer_res_norm_capped/wav2vec2-large-xlsr_{distortion_type}_results_cer.json"
    dataset_path= f"../ted3test_distorted/{distortion_type}"
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
