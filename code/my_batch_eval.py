from standardize_text import (standardize_reference_text,
                              clean_punctuations_transcript_owsm, clean_punctuations_transcript_whspr)
import torch
import numpy as np
# Apply audio preprocessing and clean reference text
def map_audio_and_text(batch):
    sr = batch["audio"]["sampling_rate"]
    waveform = batch["audio"]["array"]

    batch["waveform"] = waveform
    batch["sampling_rate"] = sr
    batch["transcript"] = standardize_reference_text((batch["text"]))
    return batch

# Apply batch prediction wav2vec; with attention mask
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

def map_batch_to_preds_lm(batch, model, processor, device):
    with torch.no_grad():
        # Preprocess audio
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

        # Forward pass
        logits = model(
            input_values,
            attention_mask=attention_mask
        ).logits

        logits_cpu = logits.cpu().numpy()

        try:
            if np.isnan(logits_cpu).any() or np.isinf(logits_cpu).any():
                raise ValueError("Invalid values in logits")
            predicted = processor.batch_decode(logits_cpu)
        except Exception as e:
            print(f"[LM decode failed: {e}] Falling back to greedy decoding.")
            pred_ids = torch.argmax(logits, dim=-1)
            predicted = processor.tokenizer.batch_decode(pred_ids)

    # Return only serializable fields
    return {"predicted": predicted}



def map_batch_to_preds_owsm(batch, s2t):
    batch_preds = []
    for waveform in batch["waveform"]:
        waveform = np.array(waveform, dtype=np.float32)
        try:
            result = s2t(
                waveform,
                lang_sym="<eng>",
                task_sym="<asr>",
                text_prev="worcestershire sauce"  # optional
            )
            pred_text_raw = result[0][-2]
        except Exception as e:
            print(f"Error during decoding: {e}")
            pred_text_raw = ""
        cleaned_pred = clean_punctuations_transcript_owsm(standardize_reference_text(pred_text_raw))
        batch_preds.append(cleaned_pred)
    batch["predicted"] = batch_preds
    return batch

def map_batch_to_preds_whisper(batch, model, processor, device, forced_decoder_ids=None):
    with torch.no_grad():
        # Convert waveforms to input features (log-mel spectrograms)
        inputs = processor.feature_extractor(
            batch["waveform"], sampling_rate=16000, return_tensors="pt", padding=True
        )
        input_features = inputs.input_features.to(device)

        # Manually create attention mask: 1 for non-zero, 0 for zero-padding
        attention_mask = (input_features != 0).long()
        input_features = inputs.input_features.to(device)

        # Generate predicted token IDs (and decode internally)
        generated_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            forced_decoder_ids=forced_decoder_ids
        )

        # Decode to strings
        batch["predicted"] = processor.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
    return batch
