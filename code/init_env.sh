# ask for interactive gpu
srun --nodes=1 --partition=gpu --qos=short --gres=gpu:1 --time=0:15:0 --account=tc068-pool2 --pty /bin/bash
export HF_HUB_OFFLINE=1 # this line will not be executed, but copied here for reference