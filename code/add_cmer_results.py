import json
import statistics
import os
from metrics_helper import char_mer

def update_results_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cmer_values = []

    # Process segments
    for seg in data.get("segments", []):
        if seg.get("wer") is None:
            seg.pop("wer", None)

        cmer_val = char_mer(seg["reference"], seg["prediction"])
        seg["cmer"] = cmer_val
        cmer_values.append(cmer_val)

    # Remove null overall_wer
    if data.get("overall_wer") is None:
        data.pop("overall_wer", None)

    if cmer_values:
        data["overall_cmer"] = char_mer(
            [seg["reference"] for seg in data["segments"]],
            [seg["prediction"] for seg in data["segments"]]
        )
        data["avg_cmer"] = round(sum(cmer_values) / len(cmer_values), 4)
        data["median_cmer"] = statistics.median(cmer_values)

        if "avg_cer" in data:
            data["avg_cer"] = round(data["avg_cer"], 4)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Updated JSON saved: {path}")

def process_directory(dir_root):
    for fname in os.listdir(dir_root):
        if fname.endswith(".json"):
            full_path = os.path.join(dir_root, fname)
            try:
                update_results_json(full_path)
            except Exception as e:
                print(f"Skipping {fname}: {e}")

if __name__ == "__main__":
    default_root = "../cer_res_norm_capped/base_comparison"
    process_directory(default_root)
