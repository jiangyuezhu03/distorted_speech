import sys, os
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["WANDB_DISABLED"] = "true" # was in the last run, uncomment to avoid warning in future runs

import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from datasets import load_from_disk, DatasetDict
from standardize_text import standardize_reference_text
import numpy as np
from jiwer import cer, wer, mer
# from nemo_text_processing.text_normalization.normalize import Normalizer
from data_collator import DataCollatorCTCWithPadding
# wav2vec does not need lm normalizer
# normalizer = Normalizer(input_case='cased', lang='en')
distortion_type = sys.argv[1] # must be fast, reversed, narrowband, or sinewave
distorted_ds = load_from_disk(f"../ted3train_5000_distorted/{distortion_type}")
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
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(
    model_name,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)).to(device)
model.train()
def prepare_dataset_from_blog(batch): # blog
    audio = batch["audio"]
    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

def prepare_dataset(batch):
    audio = batch["audio"]
    inputs = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
        padding=True
    )
    batch["input_values"] = inputs.input_values[0].numpy()

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"].lower()).input_ids

    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names, num_proc=4)
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    pred_norm = [p.strip() for p in pred_str]
    label_norm = [standardize_reference_text(l.strip()) for l in label_str]

    return {"cer": 100 * cer(label_norm, pred_norm)}

lr=5e-5
metric_for_best='cer'
# Create and use a proper output path
output_path = f"/work/tc068/tc068/jiangyue_zhu/.cache/ft/wav2vec2-large-xlsr_{distortion_type}_{metric_for_best}_{lr}"
os.makedirs(output_path, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_path,
    group_by_length=True,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=2,
    eval_strategy="steps",
    num_train_epochs=30,
    fp16=True,
    save_steps=200,
    eval_steps=200,
    logging_steps=25,
    learning_rate=lr,
    warmup_steps=500,
    max_steps=3000,
    save_total_limit=3
)

trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor
)

print('training')
trainer.train()

trainer.save_model()
processor.save_pretrained(output_path)

print(f"model saved to {output_path}")