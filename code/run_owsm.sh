source /work/tc068/tc068/jiangyue_zhu/test_venv/owsm_ctc/bin/activate
export PYTHONPATH="/work/tc068/tc068/jiangyue_zhu/espnet:$PYTHONPATH"
export CC=gcc
export CXX=g++
python code/tedtest_owsm-ctc_baseline.py
