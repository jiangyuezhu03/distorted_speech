
# this script doesn't work anyhow: wrong env, now fixed, haven't tried
source /work/tc068/tc068/jiangyue_zhu/test_venv/espnet_new/bin/activate
#export PYTHONPATH="/work/tc068/tc068/jiangyue_zhu/espnet:$PYTHONPATH"
#export CC=gcc
#export CXX=g++
export HF_HUB_OFFLINE=1
python owsm-ctc_baseline.py
