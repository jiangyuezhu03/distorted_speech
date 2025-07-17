from standardize_text import standardize_reference_text
import torch
# Apply audio preprocessing and clean reference text
def map_audio_and_text(batch):
    sr = batch["audio"]["sampling_rate"]
    waveform = batch["audio"]["array"]

    batch["waveform"] = waveform
    batch["sampling_rate"] = sr
    batch["transcript"] = standardize_reference_text((batch["text"]))
    return batch

# Apply batch prediction
def map_batch_to_preds(batch, model, processor, device):
    with torch.no_grad():
        inputs = processor(
            batch["waveform"],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            logits = model(input_values, attention_mask=attention_mask).logits
        else:
            logits = model(input_values).logits

        pred_ids = torch.argmax(logits, dim=-1)
        batch["predicted"] = processor.batch_decode(pred_ids, skip_special_tokens=True)
    return batch

def map_batch_to_preds_whisper(batch, model, processor, device, forced_decoder_ids=None):
    with torch.no_grad():
        # Convert waveforms to input features (log-mel spectrograms)
        inputs = processor.feature_extractor(
            batch["waveform"],
            sampling_rate=16000,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(device)

        # Generate predicted token IDs (and decode internally)
        generated_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids
        )

        # Decode to strings
        batch["predicted"] = processor.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
    return batch
