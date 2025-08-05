model_type = sys.argv[1]  # "base" or "ft"

if model_type == "base":
    if len(sys.argv) < 3:
        raise ValueError("Usage: base <distortion_type> [condition]")
    distortion_type = sys.argv[2]
    condition = sys.argv[3] if len(sys.argv) > 3 else None
    model_name = base_model_path
    model_identifier = "wav2vec2-large-xlsr"

elif model_type == "ft":
    if len(sys.argv) < 4:
        raise ValueError("Usage: ft <trained_on_distortion> <lr> [condition]")
    trained_on_distortion = sys.argv[2]
    lr = sys.argv[3]
    condition = sys.argv[4] if len(sys.argv) > 4 else None
    model_name = f"/work/tc068/tc068/jiangyue_zhu/.cache/ft/wav2vec2-large-xlsr_{trained_on_distortion}_cer_{lr}"
    model_basename = model_name.split("/")[-1]
    parts = model_basename.split("_")
    ft_details = '_'.join(parts[1:])
    model_identifier="".join(parts[:1]) # wav2vec2-large-xlsr
    print(f"model_identifier: {model_identifier}")
    model_output_identifier= f"{model_identifier}_{ft_details}"
    print("output identifier ",model_output_identifier)

    if "_" not in trained_on_distortion:
        distortion_type = trained_on_distortion
    else:
        distortion_type=trained_on_distortion.split("_")[0]
    print(f"distortions {distortion_type}")
else:
    raise ValueError("First argument must be either 'base' or 'ft'")
# Dataset path
if condition:
    dataset_path = f"../ted3test_distorted_adjusted/{distortion_type}_adjusted/{distortion_type}_{condition}"
    print(f"distortion_type: {distortion_type}, condition: {condition}")
    # Avoid repeating distortion_type in output file if condition already includes it
    output_path = f"/work/tc068/tc068/jiangyue_zhu/cer_res_norm_capped/{model_output_identifier}_{distortion_type}_{condition}_results.json"
else:
    dataset_path = f"../ted3test_distorted/{distortion_type}"
    output_path = f"/work/tc068/tc068/jiangyue_zhu/cer_res_norm_capped/{model_output_identifier}_{distortion_type}_results.json"

print(f"model name: {model_name}")
print(f"datapath: {dataset_path}")
print(f"output: {output_path}")