from transformers import TrainingArguments
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

device = "cuda" if torch.cuda.is_available() else "cpu"
wav2vec_lm = "/work/tc068/tc068/jiangyue_zhu/.cache/huggingface/hub/models--patrickvonplaten--wav2vec2-base-100h-with-lm/snapshots/0612413f4d1532f2e50c039b2f014722ea59db4e"
wav2vec_xlsr= "/work/tc068/tc068/jiangyue_zhu/.cache/huggingface/hub/models--jonatasgrosman--wav2vec2-large-xlsr-53-english/snapshots/569a6236e92bd5f7652a0420bfe9bb94c5664080"
wav2vec_lm_model=Wav2Vec2ForCTC.from_pretrained(wav2vec_lm).to(device).eval()
wav2vec_xlsr_model=Wav2Vec2ForCTC.from_pretrained(wav2vec_xlsr).to(device).eval()

trainable_params = sum(p.numel() for p in wav2vec_lm_model.parameters() if p.requires_grad)
print(f"Trainable parameters in wav2vec_lm: {trainable_params}") #94396320
trainable_params2 = sum(p.numel() for p in wav2vec_xlsr_model.parameters() if p.requires_grad)
print(f"Trainable parameters in wav2vec_xlsr: {trainable_params2}") # 315472545
# train_a = torch.load("../.cache/ft/whisper-small_enc_narrowband_cer_5e-05/checkpoint-1200/training_args.bin")
# train_b = torch.load("../.cache/ft/whisper-small_enc_reversed_cer_5e-05/training_args.bin")
# dict1 = vars(train_a)
# dict2 = vars(train_b)
# # Compare TrainingArguments subset
# for key in dict1:
#     if key == "output_dir":
#         continue
#     if dict1[key] != dict2[key]:
#         print(f"Difference in {key}: {dict1[key]} vs {dict2[key]}")


