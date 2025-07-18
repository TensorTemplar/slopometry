"""Hugging Face dataset upload functionality."""

from pathlib import Path

from slopometry.core.settings import settings


def upload_to_huggingface(file_path: Path, repo_id: str, token: str | None = None) -> None:
    """Upload dataset to Hugging Face Hub.

    Args:
        file_path: Path to the parquet dataset file
        repo_id: HuggingFace dataset repository ID (e.g., 'username/dataset-name')
        token: Optional HuggingFace token (defaults to settings or HF_TOKEN env var)
    """
    # Use token from settings if not provided
    if token is None and settings.hf_token:
        token = settings.hf_token
    try:
        import pandas as pd
        from datasets import Dataset, DatasetDict
        from huggingface_hub import HfApi, create_repo
    except ImportError as e:
        raise ImportError(
            "Required libraries not installed. Install with:\npip install datasets huggingface-hub pandas pyarrow"
        ) from e

    # Load the dataset from parquet
    df = pd.read_parquet(file_path)
    dataset = Dataset.from_pandas(df)

    # Create dataset dict with train split
    dataset_dict = DatasetDict({"train": dataset})

    # Create repository if it doesn't exist
    api = HfApi(token=token)
    try:
        create_repo(repo_id=repo_id, token=token, repo_type="dataset", exist_ok=True)
    except Exception:
        pass  # Repo might already exist

    # Push to hub
    dataset_dict.push_to_hub(repo_id, token=token, commit_message="Upload slopometry user story dataset")

    # Update dataset card with metadata
    dataset_card = f"""---
dataset_info:
  features:
  - name: id
    dtype: string
  - name: created_at
    dtype: timestamp[ns]
  - name: base_commit
    dtype: string
  - name: head_commit
    dtype: string
  - name: diff_content
    dtype: string
  - name: stride_size
    dtype: int64
  - name: user_stories
    dtype: string
  - name: rating
    dtype: int64
  - name: guidelines_for_improving
    dtype: string
  - name: model_used
    dtype: string
  - name: prompt_template
    dtype: string
  - name: repository_path
    dtype: string
  splits:
  - name: train
    num_examples: {len(dataset)}
  download_size: {file_path.stat().st_size}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
license: mit
task_categories:
- text-generation
language:
- en
tags:
- slopometry
- userstorify
---

# Slopometry User Story Dataset

This dataset contains git diffs paired with AI-generated user stories, collected using [slopometry](https://github.com/TensorTemplar/slopometry).

## Dataset Structure

Each entry contains:
- `base_commit`: Base commit reference
- `head_commit`: Head commit reference  
- `diff_content`: Git diff between commits
- `user_stories`: AI-generated user stories from the diff
- `rating`: Quality rating (1-5)
- `guidelines_for_improving`: Human feedback for improvement
- `model_used`: AI model used for generation
- `prompt_template`: Template used for prompting
- `repository_path`: Source repository
- `created_at`: Timestamp of generation

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
```

## Collection

Generated using `slopometry userstorify` command.
"""

    # Update README
    api.upload_file(
        path_or_fileobj=dataset_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message="Update dataset card",
    )
