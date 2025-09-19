import os
# 修改为镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

download_dir = './'

# download probs and embeddings from Hugging Face
repo_id =  'coldchair16/CPRet-Embeddings'
snapshot_download(
    repo_id=repo_id,
    repo_type='dataset',
    local_dir=os.path.join(download_dir),
    allow_patterns=['probs_2507*'],
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=4,
)
print(f"Finished downloading {repo_id} to {download_dir}")


# download models from Hugging Face
# download_dir = './'
# repo_id_list = [
#     'coldchair16/CPRetriever-Prob-Qwen3-4B',
# ]
# for repo_id in repo_id_list:
#     model_name = repo_id.split("/")[-1]
#     # replace '.' with 'p' to avoid issues in SentenceTransformer
#     model_name = model_name.replace(".", "p")
#     print('Begin downloading:', repo_id)
#     snapshot_download(
#         repo_id=repo_id,
#         repo_type="model",
#         local_dir=os.path.join(download_dir, model_name),
#         allow_patterns=['*'],
#         local_dir_use_symlinks=False,
#         resume_download=True,
#         max_workers=4,
#     )
#     print(f"Finished downloading {repo_id} to {download_dir}/{model_name}")