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
eval_logs = {}

for model_name, filename in latest_files.items():
    with open(filename, "r") as f:
        log_entries = []
        eval_log_entries = []
        for line in f:
            try:
                line_dict = ast.literal_eval(line.strip())
                if isinstance(line_dict, dict) and "loss" in line_dict:
                    log_entries.append(line_dict)
                if isinstance(line_dict, dict) and "eval_loss" in line_dict:
                    eval_log_entries.append(line_dict)
            except:
                continue
        logs[model_name] = log_entries
        eval_logs[model_name] = eval_log_entries

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# Plot training loss
for model_name, log_entries in logs.items():
    epochs = [entry["epoch"] for entry in log_entries]
    losses = [entry["loss"] for entry in log_entries]
    axes[0].plot(epochs, losses, label=model_name)

axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)
axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
# Plot evaluation loss
for model_name, eval_log_entries in eval_logs.items():  # <- fix here
    epochs = [entry["epoch"] for entry in eval_log_entries]
    losses = [entry["eval_loss"] for entry in eval_log_entries]
    axes[1].plot(epochs, losses, label=model_name)

axes[1].set_title("Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
plt.tight_layout()
plt.savefig("combined_loss_plot.jpg")
print("Saved combined plot")
plt.show()

