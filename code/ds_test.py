from datasets import load_dataset
import datasets

ds = load_dataset("LIUM/tedlium", "release3", split="test")
ds.save_to_disk("ted3test_cache")
# Try printing the waveform from the first entry
