import numpy as np
import torch
import torchaudio
import json
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer
from datasets import load_dataset


model_name = "openai/whisper-small"
output_path = "whspr-small_baseline_results.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'using {device}')
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device).eval()
# make sure uses Eng
forced_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
model.config.forced_decoder_ids = forced_ids


subset = load_dataset("LIUM/tedlium", "release3", split="test",trust_remote_code = True)
print("loaded dataset")

results = {"segments": [], "overall_wer": 0.0}
predictions = []
references = []
print("processing dataset")
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
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(
            torch.from_numpy(waveform)).numpy()
        sr = 16000


    # Generate prediction
    input_features = processor.feature_extractor(
        waveform, sampling_rate=sr, return_tensors="pt").input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_ids)

    pred_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].lower().strip()

    # Save both transcript and prediction
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
