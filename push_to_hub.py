from huggingface_hub import HfApi
import sys
api = HfApi()
# api.upload_large_folder(
#     folder_path="ted3test_distorted",
#     repo_id="jiangyuezhu/ted3test_distorted",
#     repo_type="dataset",
#     # commit_message="Initial upload of TED train distorted"
# )

from huggingface_hub import HfApi
distortion=sys.argv[1]
# Define repo info
user_or_org = "jiangyuezhu"
dataset_name = f"ted3test_{distortion}"
repo_id = f"{user_or_org}/{dataset_name}"
local_folder_path = f"ted3test_distorted/{distortion}"  # path to local dataset folder

# Instantiate API
api = HfApi()

api.create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    exist_ok=True  # prevents error if the dataset already exists
)

api.upload_folder(
    folder_path=local_folder_path,
    repo_id=repo_id,
    repo_type="dataset"
)

print(f"Dataset uploaded to: https://huggingface.co/datasets/{repo_id}")
