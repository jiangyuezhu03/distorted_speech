#espnet/owsm_v4_base_102M
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
from jiwer import cer
from tqdm import tqdm
import sys

from nemo_text_processing.text_normalization.normalize import Normalizer
normalizer = Normalizer(input_case='cased', lang='en')

# not calculating cer yet, needs text standardization
distortion_type=sys.argv[1] # must match folder names
condition = sys.argv[2] if len(sys.argv)>2 else None
if condition: # with normalizer, now save to processed folder directly
    output_path = f"/work/tc068/tc068/jiangyue_zhu/cer_res_norm_capped/owsm4_{distortion_type}_{condition}_results_cer.json"
    dataset_path= f"../ted3test_distorted_adjusted/{distortion_type}_adjusted/{distortion_type}_{condition}"
else: # not using "adjusted"
    output_path = f"/work/tc068/tc068/jiangyue_zhu/cer_res_norm_capped/owsm4_{distortion_type}_results_cer.json"
    dataset_path= f"../ted3test_distorted/{distortion_type}"
# or espnet/owsm_ctc_v4_1B, espnet/owsm_ctc_v3.1_1B

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

s2t = Speech2Text.from_pretrained(
    model_tag="espnet/owsm_v4_medium_1B",
    device=device,
    beam_size=6,          # used to be 5
    ctc_weight=0.0,       # Pure attention decoder (as in Whisper)
    maxlenratio=0.0,      # Let the model decide max decoding length
    lang_sym="<eng>",
    task_sym="<asr>",
)

print("Installed OWSM encoder-decoder pipeline")

print("Processing dataset")
subset = load_from_disk(dataset_path)
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
cer_list=[]
cer_list_capped=[]
results = {
    "segments": [],
    "overall_cer": None
}

references = []
predictions = []
for ref, hyp in zip(result["transcript"], result["predicted"]):
    pred_text = normalizer.normalize(hyp) # ref and hyp is already cleaned
    references.append(ref)
    predictions.append(pred_text)
    cer_score=cer(ref,pred_text)
    results["segments"].append({
        "reference": ref,
        "prediction": pred_text,
        "cer": cer_score
    })
    cer_list_capped.append(min(cer_score, 1.0))
    cer_list.append(cer_score)

# Overall WER
results["overall_cer"] = cer(references, predictions)
results["avg_cer"] = np.mean(cer_list_capped)
results["median_cer"] = np.median(cer_list)

# Save to JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved results to {output_path}")

