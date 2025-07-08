# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
# import torch
# import torchaudio
# import re
#
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
#
# audio_file = "../hvd_481.wav"
# waveform, sample_rate = torchaudio.load(audio_file)
#
# resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
# waveform = resampler(waveform)
#
# inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
#
# with torch.no_grad():
#   logits = model(inputs.input_values).logits
#
# predicted_ids = torch.argmax(logits, dim=-1)
# transcription = processor.decode(predicted_ids[0])
#
# print(transcription)

import numpy as np
import torch
import torchaudio
import json
from datasets import load_from_disk
from tqdm import tqdm
import sys

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from standardize_text import standardize_reference_text  # your own function

distortion_type = sys.argv[1]  # e.g., clean / fast / reversed / ...
output_path = f"/work/tc068/tc068/jiangyue_zhu/res/wav2vec2-base_{distortion_type}_results.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()

# Load dataset
dataset_path = f"/work/tc068/tc068/jiangyue_zhu/ted3test_distorted/{distortion_type}"
subset = load_from_disk(dataset_path)
print("loaded dataset")

results = {"segments": [], "overall_wer": 0.0}
predictions = []
references = []

print("processing dataset")
for sample in tqdm(subset, desc="processing dataset"):
    raw_transcript = sample['text']
    clean_transcript = standardize_reference_text(raw_transcript)

    # Skip segments not intended for scoring
    if "ignore_time_segment_in_scoring" in clean_transcript:
        continue

    audio = sample["audio"]
    waveform = audio["array"]
    sr = audio["sampling_rate"]

    # Resample to 16 kHz if needed
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(
            torch.from_numpy(waveform)
        ).numpy()
        sr = 16000

    # Prepare input
    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    # currently pred_text is raw prediction
    pred_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].lower().strip()

    predictions.append(pred_text)
    references.append(clean_transcript)

    results["segments"].append({
        "reference": clean_transcript,
        "prediction": pred_text,
        "wer": None
    })

# WER is left as None — you can compute later if needed
results["overall_wer"] = None

# Save results
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved results to {output_path}")
