import numpy as np
import torch
import torchaudio
import json
from standardize_text import standardize_reference_text
from transformers import AutoProcessor, WavLMForCTC
from datasets import load_from_disk
from my_batch_eval import map_audio_and_text, map_batch_to_preds
import sys
from jiwer import cer
from tqdm import tqdm

distortion_type=sys.argv[1] # must match folder names
condition=sys.argv[2]
# output_path = f"../res/cer_res/wavlm-large_{distortion_type}_{condition}_results.json"
output_path = f"../res/cer_res/wavlm-large_{distortion_type}_{condition}_results.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")

# Load model and processor: processor throws an error
model_name = "patrickvonplaten/wavlm-libri-clean-100h-large"
# You can try "wavlm-base-plus" or anjulRajendraSharma/WavLm-base-en
processor = AutoProcessor.from_pretrained(model_name, use_safetensors=True)#local_files_only = True,
model = WavLMForCTC.from_pretrained(model_name,  use_safetensors=True).to(device).eval()
# subset = load_from_disk(f"../ted3test_distorted/{distortion_type}")
dataset_path = f"../ted3test_distorted_adjusted/{distortion_type}_adjusted/{distortion_type}_{condition}"
subset = load_from_disk(dataset_path)
print("loaded dataset")

results = {"segments": [], "overall_wer": None}
predictions = []
references = []
print("processing dataset")

# for sample in tqdm(subset,desc="processing dataset"):
#     raw_transcript = sample['text']
#     clean_transcript = standardize_reference_text(raw_transcript)

#     # Skip non-speech markers
#     if "ignore_time_segment_in_scoring" in clean_transcript:
#         continue

#     audio = sample["audio"]
#     waveform = audio["array"]
#     sr = audio["sampling_rate"]

#     # Resample to 16kHz if needed
#     if sr != 16000:
#         waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(
#             torch.from_numpy(waveform)).numpy()
#         sr = 16000

#     # Tokenize and run model
#     inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
#     input_values = inputs.input_values.to(device)

#     with torch.no_grad():
#         logits = model(input_values).logits

#     predicted_ids = torch.argmax(logits, dim=-1)
#     pred_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].lower().strip()

#     predictions.append(pred_text)
#     references.append(clean_transcript)

#     sentence_wer = None
#     results["segments"].append({
#         "reference": clean_transcript,
#         "prediction": pred_text,
#         "cer": None
#     })

# # Overall WER
# results["overall_cer"] = None

# # Save to JSON
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(results, f, indent=2, ensure_ascii=False)

# print(f"Saved results to {output_path}")

def predict_batch(batch):
    return map_batch_to_preds(batch, model, processor, device)

# Main block just handles argument parsing and I/O
if __name__ == "__main__":
    distortion_type = sys.argv[1]
    # condition = sys.argv[2]

    # output_path = f"/work/tc068/tc068/jiangyue_zhu/res/cer_res/wav2vec2-base-lm_{distortion_type}_{condition}_results.json"
    # print("loaded model")
    # Load TED-LIUM dataset

    # dataset_path = f"../ted3test_distorted_adjusted/{distortion_type}_adjusted/{distortion_type}_{condition}"
    # subset = load_from_disk(dataset_path)

    # first try on all 8 distortions
    # subset = load_from_disk(f"../ted3test_distorted/{distortion_type}")
    print("loaded dataset")
    subset = subset.map(map_audio_and_text)
    print("mapping")

    result = subset.map(
        predict_batch,
        batched=True,
        batch_size=16,
        remove_columns=["audio", "text", "waveform", "sampling_rate"],
        num_proc=1,
        load_from_cache_file=False
    )
    print("generating results")

    results = {
        "segments": [],
        "overall_cer": None
    }

    for ref, hyp in zip(result["transcript"], result["predicted"]):
        cer_score = cer(ref, hyp)
        results["segments"].append({
            "reference": ref,
            "prediction": hyp,
            "cer": cer_score
        })

    results["overall_cer"] = cer(result["transcript"], result["predicted"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {output_path}")
