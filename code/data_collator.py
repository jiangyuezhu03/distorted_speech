from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

class WhisperDataCollatorWithAttentionMask:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [f["input_features"][0] for f in features]
        batch = self.processor.feature_extractor.pad(
            {"input_features": input_features},
            return_tensors="pt"
        )
        batch["labels"] = torch.tensor([f["labels"] for f in features])

        # Compute attention mask manually
        batch["attention_mask"] = (batch["input_features"] != 0).long()
        return batch
