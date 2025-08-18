import torch, random, numpy as np
from datasets import load_from_disk
from transformers import WhisperModel, WhisperProcessor
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import pandas as pd

outpath="../ft_layer_comparison"
os.makedirs(outpath, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model_reversed = WhisperModel.from_pretrained(
    "../.cache/ft/whisper-small_enc_reversed_cer_5e-05/checkpoint-1200").to(device).eval()
model_sinewave = WhisperModel.from_pretrained(
    "../.cache/ft/whisper-small_enc_sinewave_cer_5e-05/checkpoint-1200").to(device).eval()
model_narrowband = WhisperModel.from_pretrained(
    "../.cache/ft/whisper-small_enc_narrowband_cer_5e-05/checkpoint-1200").to(device).eval()
model_narrowband_23 = WhisperModel.from_pretrained(
    "../.cache/ft/whisper-small_enc_narrowband_mid_only_2_3_cer_5e-05/checkpoint-1600").to(device).eval()
model_fast = WhisperModel.from_pretrained(
    "../.cache/ft/whisper-small_enc_fast_cer_5e-05/checkpoint-1600").to(device).eval()
model_baseline=WhisperModel.from_pretrained("openai/whisper-small").to(device).eval()

# Make sure each example exposes raw waveform at 16k
def map_audio(batch):
    # HF audio column yields {"array": np.ndarray, "sampling_rate": int}
    sr = batch["audio"]["sampling_rate"]
    wav = batch["audio"]["array"]
    if sr != 16000:
        # if your dataset isn't already 16k, resample upstream; processor will *not* resample
        raise ValueError(f"Expected 16kHz, got {sr}")
    batch["waveform"] = wav
    return batch

#mapping is successfull
# 3) Utilities
@torch.no_grad()
def encode_hidden_states(model: WhisperModel, batch_waveforms):
    """
    batch_waveforms: list of 1D numpy arrays (variable length)
    Returns: list of tensors [ (B, T_i, H) per layer i ], from embedding through last layer.
    """
    # feats = processor(batch_waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
    feats = processor(
        batch_waveforms,
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",  # <- pads mel to 3000 frames
        return_attention_mask=True
    ).to(device)
    input_features = feats.input_features.to(device)  # (B, n_mels, n_frames)

    enc_out = model.encoder(input_features=input_features, output_hidden_states=True)
    # Tuple: (embed, layer1, ..., layerN); each shape (B, T, H) after encoder
    return [h for h in enc_out.hidden_states]

def mean_cosine(a, b):
    # a,b: (B, T, H) -> pool over time to (B, H), then mean cosine over batch
    a_pool = a.mean(dim=1)
    b_pool = b.mean(dim=1)
    return torch.nn.functional.cosine_similarity(a_pool, b_pool, dim=-1).mean().item()

def batched(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

def compare_models(
    model_a, name_a: str,
    model_b, name_b: str,
    subset,
    batch_size=16,
    distortion=None
):
    first_batch = True
    count_batches = 0
    L = None
    sum_matrix = None
    sum_diag = None

    for batch in batched(subset["waveform"], batch_size):
        H_a = encode_hidden_states(model_a, batch)
        H_b = encode_hidden_states(model_b, batch)
        if L is None:
            L = len(H_a)
            sum_matrix = np.zeros((L, L))
            sum_diag = np.zeros(L)

        mat = np.zeros((L, L))
        diag = np.zeros(L)
        for i in range(L):
            for j in range(L):
                mat[i, j] = mean_cosine(H_a[i], H_b[j])
            diag[i] = mat[i, i]  # same-layer similarity

        sum_matrix += mat
        sum_diag += diag
        count_batches += 1

    # Average across batches
    avg_matrix = sum_matrix / count_batches
    avg_diag = sum_diag / count_batches
    df_matrix = pd.DataFrame(avg_matrix, index=range(L), columns=range(L))
    df_matrix.to_csv(os.path.join(outpath, f"layer_matrix_{name_a}_vs_{name_b}_on_{distortion}.csv"), float_format="%.4f")

    df_diag = pd.DataFrame(avg_diag).T  # single row
    df_diag.to_csv(os.path.join(outpath,f"same_layer_{name_a}_vs_{name_b}_on{distortion}.csv"), float_format="%.4f", index=False)

    print("calculated average")
    # ---- Plot full layer × layer heatmap ----
    plt.figure(figsize=(6, 5))
    sns.heatmap(avg_matrix, cmap="coolwarm", vmin=-1, vmax=1,
                xticklabels=range(L), yticklabels=range(L), annot=True if L <= 12 else False,
                fmt=".2f", annot_kws={"size": 6})
    plt.xlabel(f"{name_b} layers")
    plt.ylabel(f"{name_a} layers")

    plt.suptitle(f"Layer similarity {name_a} vs {name_b} on {distortion}")
    plt.tight_layout(rect=[0,0,1,0.95])
    fname_matrix = f"layer_matrix_{name_a}_vs_{name_b}_on_{distortion}.png"
    plt.savefig(os.path.join(outpath,fname_matrix))
    plt.close()
    print(f"Saved full layer×layer similarity heatmap -> {fname_matrix}")

    # ---- Plot diagonal/same-layer similarity ----
    plt.figure(figsize=(10, 2))
    sns.heatmap([avg_diag], annot=True, cmap="BrBG", vmin=-1, vmax=1,
                xticklabels=range(L))
    plt.xlabel("Encoder layer index (0=embedding)")
    plt.yticks([], [])
    plt.suptitle(f"Same-layer cosine similarity ({name_a} vs {name_b}) on {distortion}")
    plt.tight_layout()
    fname_diag = f"similarity_{name_a}_vs_{name_b}_on_{distortion}.png"
    plt.savefig(os.path.join(outpath,fname_diag))
    plt.close()
    print(f"Saved same-layer similarity heatmap -> {fname_diag}")


if __name__ == "__main__":
    eval_distortion=sys.argv[1] if len(sys.argv)>1 else None
    dataset_path = f"/work/tc068/tc068/jiangyue_zhu/ted3test_distorted/{eval_distortion}"
    dataset = load_from_disk(dataset_path)
    rng = random.Random(42)
    idx = rng.sample(range(len(dataset)), k=200)
    subset = dataset.select(idx)
    print(f"selected {len(subset)} samples")
    subset = subset.map(map_audio)
    # compare_models(model_narrowband_23, "narrowband 2-3",
    #                model_narrowband, "narrowband 1-3",
    #                subset, distortion="narrowband")
    # compare_models(model_sinewave, "sinewave",
    #                model_reversed, "reversed",
    #                subset, distortion="reversed")
    compare_models(model_sinewave, "sinewave",
                   model_baseline, "baseline",
                   subset, distortion="reversed")
    # compare_models(model_fast, "fast",
    #                model_reversed, "reversed",
    #                subset, distortion="reversed")

    # compare_models(model_fast, "fast",
    #                model_reversed, "reversed",
    #                subset, distortion="fast")