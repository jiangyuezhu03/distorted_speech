#espnet/owsm_v4_base_102M
import nltk
nltk.data.path.append("/work/tc068/tc068/jiangyue_zhu/nltk_data")
import numpy as np
from espnet2.bin.s2t_inference import Speech2Text
from datasets import load_from_disk
from standardize_text import clean_punctuations_transcript_owsm, standardize_reference_text
import torch
import torch.utils.checkpoint
import librosa
import json
from tqdm import tqdm
import sys

distortion_type=sys.argv[1] # must match folder names
output_path = f"/work/tc068/tc068/jiangyue_zhu/res/owsm4_{distortion_type}_results.json"
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
results = {"segments": [], "overall_wer": None}
predictions = []
references = []

print("Processing dataset")
subset = load_from_disk(f"/work/tc068/tc068/jiangyue_zhu/ted3test_distorted/{distortion_type}")

for sample in tqdm(subset, desc="processing dataset"):
    raw_transcript = sample["text"]
    clean_transcript = standardize_reference_text(raw_transcript)

    if "ignore_time_segment_in_scoring" in clean_transcript:
        continue

    audio = sample["audio"]
    waveform = np.array(audio["array"],dtype=np.float32)
    sr = audio["sampling_rate"]

    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Decode with encoder-decoder model, set lang sys parameters, check line 55 of fast
    try:
        output = s2t(
            waveform,
            lang_sym="<eng>",
            task_sym="<asr>",
            text_prev="worcestershire sauce" # added after submitting job
        )

        pred_text_raw = output[0][-2]  # Final text hypothesis (without timestamp info)
    except Exception as e:
        print(f"Decoding error: {e}")
        pred_text_raw = ""

    pred_text = clean_punctuations_transcript_owsm(standardize_reference_text(pred_text_raw))
    predictions.append(pred_text)
    references.append(clean_transcript)

    results["segments"].append({
        "reference": clean_transcript,
        "prediction": pred_text,
        "wer": None
    })

# results["overall_wer"] can be filled in later, e.g., with jiwer or evaluate
results["overall_wer"] = None


# Save to JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved results to {output_path}")