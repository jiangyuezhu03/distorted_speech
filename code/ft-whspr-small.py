import pdb
import sys, os
import torch
from transformers import WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_from_disk, DatasetDict
from standardize_text import (standardize_reference_text, clean_punctuations_transcript_whspr)
import numpy as np
from jiwer import cer, wer, mer
from nemo_text_processing.text_normalization.normalize import Normalizer
from data_collator import DataCollatorSpeechSeq2SeqWithPadding

normalizer = Normalizer(input_case='cased', lang='en')

distortion_type = sys.argv[1]
distorted_ds = load_from_disk(f"../ted3train_5000_distorted/{distortion_type}")
print(f"training on {distortion_type}")
split = 0.8

train_size = int(split * len(distorted_ds))

train_ds = distorted_ds.select(range(train_size))
val_ds = distorted_ds.select(range(train_size, len(distorted_ds)))

dataset = DatasetDict({
    "train": train_ds,
    "validation": val_ds
})

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'using {device}')

model_name = "openai/whisper-small"
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters before freeze: {trainable_params}") # 241734912

# for name, param in model.named_parameters(): # 153580800 if name start with encoder
#     if name.startswith("model.decoder"): # same 88154112 if not name start with
#         param.requires_grad = False

# # freeze decoder
for name, param in model.model.decoder.named_parameters():
    param.requires_grad = False # 88154112

model.generation_config.language = "english"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
# for debugging if whisper set it to true
model.config.use_cache = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters after freeze: {trainable_params}")
# optional but usually better to put model in train mode if training follows
model.train()

#  processor includes both tokenizer and feature extractor
processor = WhisperProcessor.from_pretrained(model_name, language="english", task="transcribe")
# print("Pad token ID:", processor.tokenizer.pad_token_id) # 50257
# print("EOS token ID:", processor.tokenizer.eos_token_id) # 50257

def prepare_dataset_old(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    transcription = batch["text"].lower()
    batch["labels"] = processor.tokenizer(transcription).input_ids
    return batch


def prepare_dataset(batch):
    audio = batch["audio"]
    # Extract input features
    input = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    )
    input_features = input.input_features[0].numpy()  # convert to numpy to store in HuggingFace dataset

    batch["input_features"] = input_features
    batch["attention_mask"] = (input_features != 0).astype(int).tolist()

    # Tokenize targets
    labels = processor.tokenizer(
        batch["text"].lower(),
        padding="max_length", # should change to longest just in case
        max_length=225,
        truncation=True
    )
    batch["labels"] = labels["input_ids"]
    return batch


dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names, num_proc=4)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
)

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # avoid in-place update
    label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    # change to process list of strings
    standard_pred = [normalizer.normalize(clean_punctuations_transcript_whspr(pred_sent.strip())) for pred_sent in pred_str]
    standard_label = [standardize_reference_text(label_sent.strip()) for label_sent in label_str]
    # print(f"label: {standard_label}","\n",f"pred: {standard_pred}")
    return {"cer": 100 * cer(standard_label,standard_pred)}


lr=5e-5
metric_for_best='cer'
# Create and use a proper output path
output_path = f"/work/tc068/tc068/jiangyue_zhu/.cache/ft/whisper-small_enc_{distortion_type}_{metric_for_best}_{lr}"
os.makedirs(output_path, exist_ok=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=output_path,
    num_train_epochs=4,# overriden by max_steps
    per_device_train_batch_size=12,
    gradient_accumulation_steps=1,
    learning_rate=lr,
    warmup_steps=200,
    max_steps=2000,
    gradient_checkpointing=False, # set to false for debugging
    fp16=True,
    eval_strategy='steps' ,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    push_to_hub=False,
    # save_total_limit=3,  # optional, to limit disk usage
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],  # FIXED this from dataset["test"]
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor  # FIXED this from processor.feature_extractor
)

print('training')
trainer.train()

trainer.save_model()
processor.save_pretrained(output_path)

print(f"model saved to {output_path}")
