import os
from glob import glob

import numpy as np
import librosa

import torch
from espnet2.bin.s2t_inference import Speech2Text
from espnet2.layers.create_adapter_fn import create_lora_adapter
import espnetez as ez

model = Speech2Text.from_pretrained(
    model_tag="espnet/owsm_v4_base_102M",  # or whatever latest model tag you're using
    device=device,
    beam_size=5,          # Beam size for decoding
    ctc_weight=0.0,       # Pure attention decoder (as in Whisper)
    maxlenratio=0.0,      # Let the model decide max decoding length
    lang_sym="<eng>",
    task_sym="<asr>",
)