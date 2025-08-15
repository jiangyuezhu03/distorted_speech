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
#
from nemo_text_processing.text_normalization.normalize import Normalizer
normalizer = Normalizer(input_case='cased', lang='en')

model = sys.argv[1]
distortion_type = sys.argv[2]
condition = sys.argv[3] if len(sys.argv) > 3 else None


model_path = f"/work/tc068/tc068/jiangyue_zhu/.cache/ft/{model}"
model_name = model

model_basename = model_name.split("_")[0]
ft_dis_type = model_name.split("_")[1]
model_basename = model_basename.replace("whisper", "whspr")
ft_details = "_".join(model_name.split("_")[1:])
print(f"ft_details {ft_details}")
model_output_identifier = f"ft-{model_basename}_{ft_details}"
print(f"output identifier {model_output_identifier}")


# Final output path
if condition:
    dataset_path = f"../ted3test_distorted_adjusted/{distortion_type}_adjusted/{distortion_type}_{condition}"
    print(f"distortion_type: {distortion_type}, condition: {condition}")
    output_path = f"/work/tc068/tc068/jiangyue_zhu/cer_res_norm_capped/{model_output_identifier}_{condition}_results_cer.json"

else:
    dataset_path = f"../ted3test_distorted/{distortion_type}"
    print(f"distortion_type: {distortion_type}")
    output_path= f"/work/tc068/tc068/jiangyue_zhu/cer_res_norm_capped/{model_output_identifier}_{distortion_type}_results_cer.json"

print(f"datapath: {dataset_path}")
print(f"output: {output_path}")
# import pdb;pdb.set_trace()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'using {device}')

# does the processor use custom checkpoint or vanilla whisper?
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device).eval()
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
