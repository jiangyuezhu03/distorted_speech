#espnet/owsm_v4_base_102M
# just fixed huggingface connection on loop script, replaced it with batch script, simply save and run
#
import nltk
nltk.data.path.append("/work/tc068/tc068/jiangyue_zhu/nltk_data")
import numpy as np
from espnet2.bin.s2t_inference import Speech2Text
from datasets import load_from_disk
from standardize_text import clean_punctuations_transcript_owsm, standardize_reference_text
from my_batch_eval import map_batch_to_preds_owsm,map_audio_and_text
import torch
import torch.utils.checkpoint
import librosa
import json
from tqdm import tqdm
import sys
# not calculating cer yet, needs text standardization
distortion_type=sys.argv[1] # must match folder names
condition = sys.argv[2]
output_path = f"/work/tc068/tc068/jiangyue_zhu/res/cer_res/owsm4_{distortion_type}_{condition}_results.json"

# output_path = f"/work/tc068/tc068/jiangyue_zhu/res/owsm4_{distortion_type}_results.json"
# or espnet/owsm_ctc_v4_1B, espnet/owsm_ctc_v3.1_1B

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

s2t = Speech2Text.from_pretrained(
    model_tag="espnet/owsm_v4_base_102M",  # or whatever latest model tag you're using
    device=device,
    beam_size=5,          # Beam size for decoding
    ctc_weight=0.0,       # Pure attention decoder (as in Whisper)
    maxlenratio=0.0,      # Let the model decide max decoding length
    lang_sym="<eng>",
    task_sym="<asr>",
)

print("Installed OWSM encoder-decoder pipeline")


print("Processing dataset")
subset = load_from_disk(f"../ted3test_distorted_adjusted/{distortion_type}_adjusted/{distortion_type}_{condition}")
subset = subset.map(map_audio_and_text)
print("mapping")

def predict_batch(batch):
    return map_batch_to_preds_owsm(batch,s2t)

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
        "prediction": hyp,
        "cer": None
    })
# Save to JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved results to {output_path}")

