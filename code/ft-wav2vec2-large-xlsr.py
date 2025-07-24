from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer


model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
model.train()

def prepare_dataset(batch):
    audio = batch["audio"]

    inputs = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
        padding="longest"
    )

    batch["input_values"] = inputs.input_values[0].numpy()

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"].lower()).input_ids

    return batch


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    pred_norm = [normalizer.normalize(p.strip()) for p in pred_str]
    label_norm = [standardize_reference_text(l.strip()) for l in label_str]

    return {"cer": 100 * cer(label_norm, pred_norm)}


training_args = TrainingArguments(
    output_dir=output_path,
    per_device_train_batch_size=8,
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    learning_rate=1e-4,
    warmup_steps=500,
    max_steps=1000,
    fp16=True,
    logging_dir=os.path.join(output_path, "logs"),
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False
)

trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor
)
