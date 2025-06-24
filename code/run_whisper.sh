source /work/tc068/tc068/jiangyue_zhu/test_venv/my_test_env/bin/activate
# Redirect Hugging Face cache to work disk
export HF_HOME="/work/tc068/tc068/jiangyue_zhu/.cache/huggingface"
# Redirect JetBrains cache to work disk
export XDG_CACHE_HOME="/work/tc068/tc068/jiangyue_zhu/jetbrains_cache"
export HF_HUB_OFFLINE=1
python code/tedtest_whspr-small_baseline.py
#python my_test_script.py