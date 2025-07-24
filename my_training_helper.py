from jiwer import cer,mer, wer
from standardize_text import standardize_reference_text, clean_punctuations_transcript_whspr
def prepare_dataset_to_ids(batch, processor, decoder_start_token_id=None):
    audio = batch["audio"]

    # Process audio and transcript together
    inputs = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=batch["transcript"],
        return_tensors="pt",
    )

    # Extract input features (log-Mel spectrograms)
    batch["input_features"] = inputs.input_features[0]

    # Extract labels (token IDs)
    labels = inputs.labels[0]

    # Optionally strip decoder start token (if your collator handles it later)
    if decoder_start_token_id is not None and labels[0] == decoder_start_token_id:
        labels = labels[1:]

    batch["labels"] = labels

    # Keep raw text for later use (e.g., in evaluation)
    batch["transcript"] = batch["transcript"]

    return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace ignored index with pad token
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and labels to text
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    cer = 100 * cer(predictions=pred_str, references=label_str)
    return {"cer": cer}
