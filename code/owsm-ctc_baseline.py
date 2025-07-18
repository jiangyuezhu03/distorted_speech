import nltk
nltk.data.path.append("/work/tc068/tc068/jiangyue_zhu/nltk_data")
import torch
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch
import numpy as np
from standardize_text import clean_punctuations_transcript_owsm, standardize_reference_text
import librosa
import json
# from jiwer import wer
from datasets import load_dataset,load_from_disk
from tqdm import tqdm
import sys
# not calculating cer yet, needs text standardization
distortion_type=sys.argv[1] # must match folder names
condition = sys.argv[2]
output_path = f"/work/tc068/tc068/jiangyue_zhu/res/cer_res/owsm-ctc4_{distortion_type}_{condition}_results.json"
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
print('installed pipeline')
# import pdb; pdb.set_trace()
results = {"segments": [], "overall_wer": 0.0}
predictions = []
references = []

print("processing dataset")
# subset = load_dataset("LIUM/tedlium", "release3", split="test",trust_remote_code = True)
subset = load_from_disk(f"/work/tc068/tc068/jiangyue_zhu/ted3test_distorted_adjusted/{distortion_type}_adjusted/{distortion_type}_{condition}")
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
    pred_text = clean_punctuations_transcript_owsm(standardize_reference_text(pred_text_raw))
    predictions.append(pred_text)
    references.append(clean_transcript)


    results["segments"].append({
        "reference": clean_transcript,
        "prediction": pred_text,
        "wer": None
    })

# Overall WER
results["overall_wer"] = None

# Save to JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved results to {output_path}")