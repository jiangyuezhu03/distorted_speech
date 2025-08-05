import nltk
nltk.data.path.append("/work/tc068/tc068/jiangyue_zhu/nltk_data")
import torch
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch
import numpy as np
from standardize_text import clean_punctuations_transcript_owsm, standardize_reference_text
import librosa
import json
from jiwer import cer
from datasets import load_dataset,load_from_disk
from tqdm import tqdm
import sys

from nemo_text_processing.text_normalization.normalize import Normalizer
normalizer = Normalizer(input_case='cased', lang='en')


distortion_type=sys.argv[1] # must match folder names
condition = sys.argv[2] if len(sys.argv)>2 else None
if condition: # with normalizer, now save to processed folder directly
    output_path = f"/work/tc068/tc068/jiangyue_zhu/cer_res_norm_capped/owsm-ctc4_{distortion_type}_{condition}_results_cer.json"
    dataset_path= f"../ted3test_distorted_adjusted/{distortion_type}_adjusted/{distortion_type}_{condition}"
else: # not using "adjusted"
    output_path = f"/work/tc068/tc068/jiangyue_zhu/cer_res_norm_capped/owsm-ctc4_{distortion_type}_results_cer.json"
    dataset_path= f"../ted3test_distorted/{distortion_type}"
# or espnet/owsm_ctc_v4_1B, espnet/owsm_ctc_v3.1_1B
context_len_in_secs = 4  # left and right context when doing buffered inference
batch_size = 32
s2t = Speech2TextGreedySearch.from_pretrained(
    "espnet/owsm_ctc_v4_1B",
    device="cuda",
    use_flash_attn=False,   # set to True for better efficiency if flash attn is installed and dtype is float16 or bfloat16
    lang_sym='<eng>',
    task_sym='<asr>',
)
print('installed owsm-ctc pipeline')
# import pdb; pdb.set_trace()
results = {"segments": [], "overall_cer": None}
predictions = []
references = []
cer_list = []
cer_list_capped=[]
print("processing dataset")

subset = load_from_disk(dataset_path)
for sample in tqdm(subset,desc="processing dataset"):
    raw_transcript = sample['text']
    clean_transcript = standardize_reference_text(raw_transcript)

    # Skip non-speech markers
    if "ignore_time_segment_in_scoring" in clean_transcript:
        continue

    audio = sample["audio"]
    waveform = audio["array"]
    sr = audio["sampling_rate"]

    # Resample if needed
    if sr != 16000:
    # use librosa to resample
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000


    pred_text_raw= s2t.decode_long_batched_buffered(
        waveform,
        batch_size=batch_size,
        context_len_in_secs=context_len_in_secs,
    )
    # owsm predictions also output extra space between apostrophe (we 're) just like the transcript
    pred_text = normalizer.normalize(clean_punctuations_transcript_owsm(standardize_reference_text(pred_text_raw)))
    predictions.append(pred_text)
    references.append(clean_transcript)
    cer_score=cer(clean_transcript,pred_text)

    results["segments"].append({
        "reference": clean_transcript,
        "prediction": pred_text,
        "cer": cer_score
    })
    cer_list_capped.append(min(cer_score,1.0))
    cer_list.append(cer_score)

results["overall_cer"] = cer(references, predictions)
results["avg_cer"] = np.mean(cer_list_capped)
results["median_cer"] = np.median(cer_list)
# Save to JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved results to {output_path}")