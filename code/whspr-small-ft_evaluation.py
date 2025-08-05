model_type = sys.argv[1]  # "base" or "ft"

if model_type == "base":
    if len(sys.argv) < 3:
        raise ValueError("Usage: base <distortion_type> [condition]")
    distortion_type = sys.argv[2]
    condition = sys.argv[3] if len(sys.argv) > 3 else None
    model_name = "openai/whisper-small"
    model_identifier = "whspr-small"

elif model_type == "ft":
    if len(sys.argv) < 4:
        raise ValueError("Usage: ft <enc|full> <trained_on_distortion> <lr> [condition]")
    training_scope = sys.argv[2]  # "enc" or "full"
    trained_on_distortion = sys.argv[3]
    lr = sys.argv[4]
    condition = sys.argv[5] if len(sys.argv) > 5 else None
    if training_scope == "enc":
        model_name = f"/work/tc068/tc068/jiangyue_zhu/.cache/ft/whisper-small_enc_{trained_on_distortion}_cer_{lr}"
        model_identifier = f"ft-whisper-small_enc_{trained_on_distortion}_cer_{lr}"
    elif training_scope == "full":
        # skip 'full' in the model path
        model_name = f"/work/tc068/tc068/jiangyue_zhu/.cache/ft/whisper-small_{trained_on_distortion}_cer_{lr}"
        model_identifier = f"ft-whisper-small_{trained_on_distortion}_cer_{lr}"
    else:
        raise ValueError("training_scope must be 'enc' or 'full'")
    # For now, assume lr is fixed
    model_name = f"/work/tc068/tc068/jiangyue_zhu/.cache/ft/whisper-small_{training_scope}_{trained_on_distortion}_cer_{lr}"
    model_basename = model_name.split("/")[-1]
    parts = model_basename.split("_")
    model_short = parts[0].replace("whisper", "whspr")
    ft_details = '_'.join(parts[1:])  # everything after 'whisper-small'
    print(f"details {ft_details}")
    model_output_identifier = f"ft-{model_short}_{ft_details}"
    print(f"output identifier {model_output_identifier}")

    # Evaluation distortion type is same as trained-on unless you change it
    if "_" not in trained_on_distortion:
        distortion_type = trained_on_distortion
    else:
        distortion_type=trained_on_distortion.split("_")[0]
    print(f"distortions {distortion_type}")

else:
    raise ValueError("First argument must be either 'base' or 'ft'")

# Final output path
if condition:
    dataset_path = f"../ted3test_distorted_adjusted/{distortion_type}_adjusted/{distortion_type}_{condition}"
    print(f"distortion_type: {distortion_type}, condition: {condition}")
    # Avoid repeating distortion_type in output file if condition already includes it
    if model_type == "ft":
        output_path = f"/work/tc068/tc068/jiangyue_zhu/res/cer_res/{model_output_identifier}_{condition}_results.json"
    else:
        output_path = f"/work/tc068/tc068/jiangyue_zhu/res/cer_res/{model_identifier}_{distortion_type}_{condition}_results.json"
else:
    dataset_path = f"../ted3test_distorted/{distortion_type}"
    output_path = f"/work/tc068/tc068/jiangyue_zhu/res/cer_res/{model_output_identifier}_{distortion_type}_results.json"

print(f"model name: {model_name}")
print(f"datapath: {dataset_path}")
print(f"output: {output_path}")
# import pdb;pdb.set_trace()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'using {device}')