# ask for interactive gpu
srun --nodes=1 --partition=gpu --qos=short --gres=gpu:1 --time=0:10:0 --account=tc068-pool2 --pty /bin/bash