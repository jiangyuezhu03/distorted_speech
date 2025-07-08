import numpy as np
from standardize_text import standardize_reference_text, clean_punctuations_transcript_whspr
import torch
import torchaudio
import json
from transformers import WhisperProcessor, WhisperForConditionalGeneration
# from jiwer import wer
from datasets import load_dataset,load_from_disk
from tqdm import tqdm
import sys

distortion_type=sys.argv[1]

model_name = "openai/whisper-small"
output_path = f"/work/tc068/tc068/jiangyue_zhu/res/whspr-small_{distortion_type}_results.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'using {device}')
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device).eval()
# make sure uses Eng
forced_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
model.config.forced_decoder_ids = forced_ids

results = {"segments": [], "overall_wer": 0.0}
predictions = []
references = []

# subset = load_dataset("LIUM/tedlium", "release3", split="test",trust_remote_code = True)
subset = load_from_disk(f"/work/tc068/tc068/jiangyue_zhu/ted3test_distorted/{distortion_type}")

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
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(
            torch.from_numpy(waveform)).numpy()
        sr = 16000

    # Generate prediction
    input_features = processor.feature_extractor(
        waveform, sampling_rate=sr, return_tensors="pt").input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_ids)

    pred_text_raw = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    pred_text = clean_punctuations_transcript_whspr(pred_text_raw)
    # Save both transcript and prediction
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
