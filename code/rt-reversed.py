import torch.nn as nn

reverse_win_ms = 20 # 20, 31, 62
sr = 16000
kernel_size = int(sr * (reverse_win_ms / 1000.0))  # e.g., 320 samples
conv = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=kernel_size, stride=kernel_size)

# add to model or experiment in a custom wrapper model
