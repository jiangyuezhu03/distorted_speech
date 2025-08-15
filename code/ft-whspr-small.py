import sys, os
import torch
from transformers import WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
from datasets import load_from_disk, DatasetDict
from standardize_text import (standardize_reference_text, clean_punctuations_transcript_whspr)
import numpy as np
from jiwer import cer, wer, mer
from nemo_text_processing.text_normalization.normalize import Normalizer
from data_collator import DataCollatorSpeechSeq2SeqWithPadding
from ft_helper import find_latest_checkpoint

normalizer=Normalizer(input_case='cased', lang='en')
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

#  Checkpoint detection
lr = 5e-5
metric_for_best = 'cer'
output_path = f"/work/tc068/tc068/jiangyue_zhu/.cache/ft/whisper-small_enc_{distortion_type}_{metric_for_best}_{lr}"
os.makedirs(output_path, exist_ok=True)

# latest_checkpoint = find_latest_checkpoint(output_path)
model_name = "openai/whisper-small"

# if latest_checkpoint:
#     print(f"Found checkpoint: {latest_checkpoint}, loading checkpoint model")
#     model = WhisperForConditionalGeneration.from_pretrained(latest_checkpoint, use_safetensors=True).to(device)
# else:
#     print("Did not found checkpoint, starting from scratch")
model = WhisperForConditionalGeneration.from_pretrained(model_name, use_safetensors=True).to(device)

#  Freeze decoder
print(f"Trainable parameters before freeze: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
for name, param in model.model.decoder.named_parameters():
    param.requires_grad = False
print(f"Trainable parameters after freeze: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

#  Model config
processor = WhisperProcessor.from_pretrained(model_name, language="english", task="transcribe")

model.generation_config.language = "english"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.use_cache = False
print("initialized model")

# Dataset preprocessing
def prepare_dataset(batch):
    audio = batch["audio"]
    input = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    )
    input_features = input.input_features[0].numpy()
    batch["input_features"] = input_features
    batch["attention_mask"] = (input_features != 0).astype(int).tolist()
    labels = processor.tokenizer(
        batch["text"].lower(),
        padding="max_length",
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
    label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    standard_pred = [normalizer.normalize(clean_punctuations_transcript_whspr(p.strip())) for p in pred_str]
    standard_label = [standardize_reference_text(l.strip()) for l in label_str]
    return {"cer": 100 * cer(standard_label, standard_pred)}

training_args = Seq2SeqTrainingArguments(
    output_dir=output_path,
    num_train_epochs=4,
    per_device_train_batch_size=12,
    gradient_accumulation_steps=2,
    learning_rate=lr,
    warmup_steps=300,
    max_steps=3000,
    gradient_checkpointing=False,
    fp16=True,
    eval_strategy='steps',
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=400,
    eval_steps=200,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    push_to_hub=False,
    save_total_limit=3,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    tokenizer=processor
)
print("training")
# if latest_checkpoint:
#     print(f"Resuming training from checkpoint: {latest_checkpoint}")
#     trainer.train(resume_from_checkpoint=latest_checkpoint)
# else:
#     print("Starting new training")
trainer.train()

trainer.save_model()
processor.save_pretrained(output_path)
print(f"Model saved to {output_path}")
