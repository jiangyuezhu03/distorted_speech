from transformers import AutoProcessor, Wav2Vec2ForCTC, WavLMForCTC

model_name = "facebook/wav2vec2-base-10k-voxpopuli-ft-en"  # or the WavLM one

try:
    model_path = "/work/tc068/tc068/jiangyue_zhu/.cache/huggingface/hub/models--facebook--wav2vec2-base-10k-voxpopuli-ft-en/snapshots/328f7961ee96d2db3af8bbd22c685f5dd96f9692"
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(model_path, local_files_only=True)
    print("success")
except Exception as e:
    import traceback
    traceback.print_exc()
