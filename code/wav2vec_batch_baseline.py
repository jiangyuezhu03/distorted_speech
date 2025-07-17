# Import section stays the same
from datasets import load_from_disk
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import torch
import sys
import json
from my_batch_eval import map_audio_and_text, map_batch_to_preds
from jiwer import cer

# 1. Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load model and processor BEFORE defining the mapping function
# model_name = "jonatasgrosman/wav2vec2-base"
model_name = "/work/tc068/tc068/jiangyue_zhu/.cache/huggingface/hub/models--patrickvonplaten--wav2vec2-base-100h-with-lm/snapshots/0612413f4d1532f2e50c039b2f014722ea59db4e"

processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name, use_safetensors=True).to(device).eval()

# 3. Now define the mapping function — it will now refer to *initialized* globals
def predict_batch(batch):
    return map_batch_to_preds(batch, model, processor, device)

# Main block just handles argument parsing and I/O
if __name__ == "__main__":
    distortion_type = sys.argv[1]
    condition = sys.argv[2]

    output_path = f"/work/tc068/tc068/jiangyue_zhu/res/cer_res/wav2vec2-base-lm_{distortion_type}_{condition}_results.json"
    print("loaded model")

    dataset_path = f"../ted3test_distorted_adjusted/{distortion_type}_adjusted/{distortion_type}_{condition}"
    subset = load_from_disk(dataset_path)
    subset = subset.map(map_audio_and_text)
    print("mapping")

    result = subset.map(
        predict_batch,
        batched=True,
        batch_size=16,
        remove_columns=["audio", "text", "waveform", "sampling_rate"]
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
