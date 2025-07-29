import re
from collections import defaultdict

# Simulated grep output
with open("model_paths.txt","r") as f:
    grep_lines = f.readlines()
# print(grep_lines)
# Group by model name, keep file with largest job ID
model_files = defaultdict(lambda: ("", -1))  # model_name: (filename, jobid)

for line in grep_lines:
    match = re.match(r"([^:]+):model saved to .*/([a-zA-Z0-9\-_]+)", line)
    if match:
        filename, model_name = match.groups()
        jobid_match = re.search(r"_(\d+)\.out", filename)
        if jobid_match:
            jobid = int(jobid_match.group(1))
            if jobid > model_files[model_name][1]:
                model_files[model_name] = (filename, jobid)

# Now you have latest .out file per model
latest_files = {model: fname for model, (fname, _) in model_files.items()}
print(len(latest_files))
import ast

logs = {}  # model_name -> list of dicts

for model_name, filename in latest_files.items():
    with open(filename, "r") as f:
        log_entries = []
        for line in f:
            try:
                line_dict = ast.literal_eval(line.strip())
                if isinstance(line_dict, dict) and "loss" in line_dict:
                    log_entries.append(line_dict)
            except:
                continue
        logs[model_name] = log_entries

import matplotlib.pyplot as plt

for model_name, log_entries in logs.items():
    epochs = [entry["epoch"] for entry in log_entries]
    losses = [entry["loss"] for entry in log_entries]
    plt.plot(epochs, losses, label=model_name)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss by Model")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"curve.jpg")
print("saved plot")
plt.show()
