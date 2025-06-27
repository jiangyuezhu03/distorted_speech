from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch
import numpy as np
import torch
import json
from jiwer import wer
from datasets import load_dataset
output_path = "owsm-ctc_results_clean.json"
# or espnet/owsm_ctc_v4_1B
context_len_in_secs = 4  # left and right context when doing buffered inference
batch_size = 32
s2t = Speech2TextGreedySearch.from_pretrained(
    "espnet/owsm_ctc_v3.1_1B",
    device="cuda",
    use_flash_attn=False,   # set to True for better efficiency if flash attn is installed and dtype is float16 or bfloat16
    lang_sym='<eng>',
    task_sym='<asr>',
)
print('installed pipeline')

results = {"segments": [], "overall_wer": 0.0}
predictions = []
references = []

print("processing dataset")
subset = load_dataset("LIUM/tedlium", "release3", split="test",trust_remote_code = True)

for sample in subset:
    transcript = sample["text"].strip().lower()

    # Skip non-speech markers
    if "ignore_time_segment_in_scoring" in transcript or "inter_segment_gap" in transcript:
        continue

    audio = sample["audio"]
    waveform = audio["array"]
    sr = audio["sampling_rate"]

    # Resample if needed
    if sr != 16000:
    # use librosa to resample
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000


    pred_text = s2t.decode_long_batched_buffered(
        waveform,
        batch_size=batch_size,
        context_len_in_secs=context_len_in_secs,
    )
    predictions.append(pred_text)
    references.append(transcript)

    # Optional: compute and log sentence-level WER
    sentence_wer = wer([pred_text], [transcript])

    results["segments"].append({
        "reference": transcript,
        "prediction": pred_text,
        "wer": round(sentence_wer, 4)
    })

# Overall WER
results["overall_wer"] = round(wer(predictions, references), 4)

# Save to JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved results to {output_path}")