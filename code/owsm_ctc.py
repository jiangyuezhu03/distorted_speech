from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch
import numpy as np
import datasets
print('successful import') # numpy and datasets are no problem
s2t = Speech2TextGreedySearch.from_pretrained(
    "espnet/owsm_ctc_v3.1_1B",
    device="cuda",
    use_flash_attn=False,   # set to True for better efficiency if flash attn is installed and dtype is float16 or bfloat16
    lang_sym='<eng>',
    task_sym='<asr>',
)

res = s2t.batch_decode(
    "sample_audio/clean/sent1.wav",    # a single audio (path or 1-D array/tensor) as input
    batch_size=16,
    context_len_in_secs=4,
)   # res is a single str, i.e., the predicted text without special tokens
print(f'single res is {res}')
res2 = s2t.batch_decode(
    ["sample_audio/clean/sent1.wav", "sample_audio/clean/sent2.wav"], # a list of audios as input
    batch_size=16,
    context_len_in_secs=4,
)   # res is a list of str

# Please check the code of `batch_decode` for all supported inputs
print(f'multiple res is {res2}')

# single res is i hope i have given you a sense of how difficult it is thank you
# multiple res is ['i hope i have given you a sense of how difficult it is thank you',
#                  'this number on average is going to go up and so overall that will more than double the services delivered per person']