# Please run this in your Google Colab Environment

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
import re

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
import pdb; pdb.set_trace()

audio_file = "hvd_481.wav"
waveform, sample_rate = torchaudio.load(audio_file)

resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
waveform = resampler(waveform)

inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)

with torch.no_grad():
  logits = model(inputs.input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])

print("transcript:", transcription)


# def clean_transcription(text):
#   filler_words = ["A", "YOU KNOW", "AM", "BASICALLY", "LIKE", "SO YA"]
#   pattern = r"\b(" + "|".join(filler_words) + r")\b"
#   return re.sub(pattern, "", text)
#
# refined_transcription = clean_transcription(transcription)
# print(refined_transcription)
