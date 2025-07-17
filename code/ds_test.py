from datasets import load_from_disk

ds = load_from_disk("/work/tc068/tc068/jiangyue_zhu/ted3test_distorted/clean")
print(ds[:2].keys())#dict_keys(['audio', 'text', 'speaker_id', 'gender', 'file', 'id'])
