import numpy as np
import torch
import torchaudio
import json
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch
from jiwer import wer
from datasets import load_dataset
s2t = Speech2TextGreedySearch.from_pretrained(
    "espnet/owsm_ctc_v3.1_1B",
    device="cuda",
    use_flash_attn=False,   # set to True for better efficiency if flash attn is installed and dtype is float16 or bfloat16
    lang_sym='<eng>',
    task_sym='<asr>',
)
print('installed pipeline')
subset = load_dataset("LIUM/tedlium", "release3", split="test",trust_remote_code = True)
print('sucessful import')
