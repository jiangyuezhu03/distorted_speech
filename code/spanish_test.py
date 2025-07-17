from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
from tqdm import tqdm
import sys
import json
from jiwer import wer
from standardize_text import standardize_reference_text,clean_punctuations_transcript_whspr

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'using {device}')

distortion_type=sys.argv[1]
output_path=f"../spanish_res/spanish_whspr-small_batch_{distortion_type}_results.json"
# Load fine-tuned model
model_name = "openai/whisper-small"  # or use huggingface ID
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device).eval()
# make sure uses Eng
forced_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
model.config.forced_decoder_ids = forced_ids

results = {"segments": [], "overall_wer": 0.0}
predictions = []
references = []

# subset = load_dataset("LIUM/tedlium", "release3", split="test",trust_remote_code = True)
full_set = load_from_disk(f"../spanish_stimuli/")
subset = [ #['Glimpsed', 'NoiseVocoded', 'Sculpted', 'SineWave', 'SpectralBand', 'TimeCompressed', 'TimeReversal', 'ToneVocoded']
    {
        "audio": sample[distortion_type]["audio"],
        "text": sample[distortion_type]["transcription"]
    }
    for sample in full_set
]
for sample in tqdm(subset,desc="processing dataset"):
    raw_transcript = sample['text']
    clean_transcript = standardize_reference_text(raw_transcript)

    audio = sample["audio"]
    waveform = audio["array"]
    sr = audio["sampling_rate"]

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
        "wer": wer(clean_transcript, pred_text)
    })

# Overall WER
results["overall_wer"] = wer(["reference"],["prediction"])

# Save to JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved results to {output_path}")
