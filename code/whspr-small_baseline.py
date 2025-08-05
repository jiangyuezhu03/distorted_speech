import numpy as np
from standardize_text import standardize_reference_text, clean_punctuations_transcript_whspr
import torch
import torchaudio
import json
from jiwer import cer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from my_batch_eval import map_audio_and_text, map_batch_to_preds_whisper
from datasets import load_dataset,load_from_disk
from tqdm import tqdm
import sys

from nemo_text_processing.text_normalization.normalize import Normalizer
normalizer = Normalizer(input_case='cased', lang='en')

distortion_type=sys.argv[1] # must match folder names
condition = sys.argv[2] if len(sys.argv)>2 else None
if condition: # with normalizer, now save to processed folder directly
    output_path = f"/work/tc068/tc068/jiangyue_zhu/cer_res_norm_capped/whspr-small_{distortion_type}_{condition}_results_cer.json"
    dataset_path= f"../ted3test_distorted_adjusted/{distortion_type}_adjusted/{distortion_type}_{condition}"
else: # not using "adjusted"
    output_path = f"/work/tc068/tc068/jiangyue_zhu/cer_res_norm_capped/whspr-small_{distortion_type}_results_cer.json"
    dataset_path= f"../ted3test_distorted/{distortion_type}"
# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name="openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device).eval()
# make sure uses Eng
forced_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
model.config.forced_decoder_ids = forced_ids

results = {"segments": [], "overall_cer": None}
predictions = []
references = []

subset = load_from_disk(dataset_path)
subset = subset.map(map_audio_and_text)
print("mapping")

def predict_batch(batch):
    return map_batch_to_preds_whisper(batch, model, processor, device,forced_ids)

result = subset.map(
        predict_batch,
        batched=True,
        batch_size=16,
        remove_columns=["audio", "text", "waveform", "sampling_rate"]
    )
print("generating results")
cer_list=[]
cer_list_capped=[]
results = {
    "segments": [],
    "overall_cer": None
}
references = []
predictions = []
for ref, hyp in zip(result["transcript"], result["predicted"]):
    pred_text=normalizer.normalize(clean_punctuations_transcript_whspr(hyp))
    references.append(ref)
    predictions.append(pred_text)
    cer_score=cer(ref,pred_text)
    results["segments"].append({
        "reference": ref,
        "prediction": pred_text,
        "cer": cer_score
    })

    cer_list_capped.append(min(cer_score, 1.0))
    cer_list.append(cer_score)

# Overall WER
results["overall_cer"] = cer(references, predictions)
results["avg_cer"] = np.mean(cer_list_capped)
results["median_cer"] = np.median(cer_list)


with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved results to {output_path}")
