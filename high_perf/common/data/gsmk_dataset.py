from torch.utils.data import DataLoader
import time
from datasets import load_dataset

# https://huggingface.co/datasets/gsm8k
HUGGING_FACE_GSMK_DATASET_ID = "gsm8k"

# Manual seed for reproducatibility

def get_data_loader(batch_size, num_workers):
    dataset = load_dataset(HUGGING_FACE_GSMK_DATASET_ID, 'main')['train']
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader
