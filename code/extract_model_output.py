from ft_helper import find_latest_checkpoint
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import json, os

def load_trainer_state(checkpoint_dir):
    ts_file = os.path.join(checkpoint_dir, "trainer_state.json")
    with open(ts_file) as f:
        state = json.load(f)
    return state.get("log_history", [])

def split_train_eval(log_history):
    train_logs = [x for x in log_history if "loss" in x and "eval_loss" not in x]
    eval_logs = [x for x in log_history if "eval_loss" in x]
    return train_logs, eval_logs

def plot_logs_sep(train_logs, eval_logs, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14,5), sharex=True)

    # Training loss
    if train_logs:
        epochs = [x["epoch"] for x in train_logs]
        losses = [x["loss"] for x in train_logs]
        axes[0].plot(epochs, losses, label=model_name)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    # Validation / evaluation loss
    if eval_logs:
        epochs = [x.get("epoch", i) for i, x in enumerate(eval_logs)]
        losses = [x["eval_loss"] for x in eval_logs]
        axes[1].plot(epochs, losses, label=model_name)
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True)

    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("combined_loss_plot.jpg")
    print("Saved combined plot")
    plt.show()
def plot_train_val_losses(model_dirs, save_path):
    cmap = cm.get_cmap("tab10", len(model_dirs))  # one base color per model
    plt.figure(figsize=(10, 6))

    for idx, model_dir in enumerate(model_dirs):
        latest_ckpt = find_latest_checkpoint(model_dir)
        if latest_ckpt is None:
            print(f"No checkpoints found for {model_dir}")
            continue
        log_history = load_trainer_state(latest_ckpt)
        train_logs, eval_logs = split_train_eval(log_history)

        model_label = os.path.basename(model_dir).replace("whisper", "whspr")

        # Base color from colormap
        base_color = cmap(idx)

        # Training = base color, Validation = brightened version
        train_color = base_color
        val_color = tuple(np.clip(np.array(base_color) + 0.3, 0, 1))  # brighten

        # Training loss
        if train_logs:
            epochs = [x["epoch"] for x in train_logs]
            losses = [x["loss"] for x in train_logs]
            plt.plot(epochs, losses, label=f"{model_label} train",
                     linestyle='-', color=train_color)

        # Validation loss
        if eval_logs:
            epochs = [x.get("epoch", i) for i, x in enumerate(eval_logs)]
            losses = [x["eval_loss"] for x in eval_logs]
            plt.plot(epochs, losses, label=f"{model_label} val",
                     linestyle='--', color=val_color)

    plt.title("Training and Validation Loss (Grouped by Model)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig(save_path)
    print("Saved combined plot")

root_path = "/work/tc068/tc068/jiangyue_zhu/.cache/ft"
with open("finetuned_models.txt") as f:
    model_names = [line.strip() for line in f if line.strip()]

model_dirs = [os.path.join(root_path, name) for name in model_names]
save_path = "/work/tc068/tc068/jiangyue_zhu/picture_results/combined_losses_by_model.png"
# for model_dir in model_dirs:
#     latest_ckpt = find_latest_checkpoint(model_dir)
#     if latest_ckpt is None:
#         print(f"No checkpoints found for {model_dir}")
#         continue
#     log_history = load_trainer_state(latest_ckpt)
#     train_logs, eval_logs = split_train_eval(log_history)
    # plot_logs(train_logs, eval_logs, os.path.basename(model_dir))
plot_train_val_losses(model_dirs,save_path)

