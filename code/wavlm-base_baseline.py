import numpy as np
import torch
import torchaudio
import json
from transformers import AutoProcessor, WavLMForCTC
from jiwer import wer
from datasets import load_dataset

output_path = "wavlm_base_results_clean.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")

# Load model and processor
model_name = "microsoft/wavlm-base"  # You can try "wavlm-base-plus" or "wavlm-large" if needed
processor = AutoProcessor.from_pretrained(model_name)
model = WavLMForCTC.from_pretrained(model_name).to(device).eval()
import pdb; pdb.set_trace()
# Load TED-LIUM dataset
subset = load_dataset("LIUM/tedlium", "release3", split="test", trust_remote_code=True)
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

    # Resample to 16kHz if needed
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(
            torch.from_numpy(waveform)).numpy()
        sr = 16000

    # Tokenize and run model
    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    pred_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].lower().strip()

    predictions.append(pred_text)
    references.append(transcript)

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
