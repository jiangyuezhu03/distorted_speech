from datasets import load_dataset

ds = load_dataset("LIUM/tedlium", "release3", split="test")

# Try printing the waveform from the first entry
print(ds[0]["audio"])  # If decoded, this is a dict with array + sampling rate
