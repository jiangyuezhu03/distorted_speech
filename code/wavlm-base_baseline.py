import numpy as np
import torch
import torchaudio
import json
from standardize_text import standardize_reference_text
from transformers import AutoProcessor, WavLMForCTC
from datasets import load_from_disk
import sys
from tqdm import tqdm

distortion_type=sys.argv[1] # must match folder names
output_path = f"/work/tc068/tc068/jiangyue_zhu/res/wavlm-base_{distortion_type}_results.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")

# Load model and processor: processor throws an error
model_name = "patrickvonplaten/wavlm-libri-clean-100h-base"
# You can try "wavlm-base-plus" or anjulRajendraSharma/WavLm-base-en
model_path = "/work/tc068/tc068/jiangyue_zhu/.cache/huggingface/hub/"
processor = AutoProcessor.from_pretrained(model_name,local_files_only=True, use_safetensors=True)
model = WavLMForCTC.from_pretrained(model_name, local_files_only=True, use_safetensors=True).to(device).eval()

# import pdb; pdb.set_trace()
# Load TED-LIUM dataset
subset = load_from_disk(f"/work/tc068/tc068/jiangyue_zhu/ted3test_distorted/{distortion_type}")
print("loaded dataset")

results = {"segments": [], "overall_wer": 0.0}
predictions = []
references = []
print("processing dataset")

for sample in tqdm(subset,desc="processing dataset"):
    raw_transcript = sample['text']
    clean_transcript = standardize_reference_text(raw_transcript)

    # Skip non-speech markers
    if "ignore_time_segment_in_scoring" in clean_transcript:
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
    references.append(clean_transcript)

    sentence_wer = None
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
