import torch
from torch.utils.data import DataLoader
import time
import cotracker.models.build_cotracker
from cotracker.datasets.tap_vid_datasets import TapVidDataset
import os 
from cotracker.datasets.utils import collate_fn
from datasets import load_dataset

### Setup ###
BATCH_SIZE = 1
BATCH_COUNT = 5
NUM_WORKERS = 1

# Manual seed for reproducatibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

DEVICE_CUDA = 'cuda'
DEVICE_CPU = 'cpu'


def get_data_loader(num_workers=1):
    dataset = load_dataset("HuggingFaceH4/no_robots")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return dataloader


def get_model(checkpoint_path=CHECKPOINT_S4_W12):
    return cotracker.models.build_cotracker.build_cotracker(checkpoint_path)


def run_inference(dataloader, model, cuda=True):
    load_time_per_batch = torch.zeros(BATCH_COUNT)
    inference_time_per_batch = torch.zeros(BATCH_COUNT)
    total_time_per_batch = torch.zeros(BATCH_COUNT)
    
    device = DEVICE_CUDA if cuda else DEVICE_CPU
    print("Working on device: {}".format(device))
    model.to(device)
    
    for batch_idx in range(BATCH_COUNT):
        print("Starting BATCHs {} of {}".format(batch_idx + 1, BATCH_COUNT))
        (output, load_time, train_time), batch_time = measure_runtime(run_batch_inference,
                                                              dataloader,
                                                              model,
                                                              cuda)
        load_time_per_batch[batch_idx] = load_time
        inference_time_per_batch[batch_idx] = train_time
        total_time_per_batch[batch_idx] = batch_time

        print("Finished Batch {} of {}".format(batch_idx + 1, BATCH_COUNT))
        print("Batch load time: {}".format(load_time))
        print("Batch inference time: {}".format(train_time))
        print("Batch total time: {}".format(batch_time))
    return model, load_time_per_batch, inference_time_per_batch, total_time_per_batch


def measure_runtime(func, *func_args):
    start = time.perf_counter()
    result = func(*func_args)
    end = time.perf_counter()
    elapsed = end - start
    return result, elapsed


def run_batch_inference(dataloader, model, cuda=True):
    (x, y), load_time = measure_runtime(
        __get_next_batch, dataloader)

    if cuda:
        x = x.to(DEVICE_CUDA)
        y = y.to(DEVICE_CUDA)

    output, train_time = measure_runtime(
        model,
        x)
    
    return output, load_time, train_time

def __get_next_batch(dataloader):
    return next(iter(dataloader))


def benchmark():
    print("Starting up...")

    print("Building data loaders...")
    data_loader = get_data_loader()

    print("Initializing Model...")
    net = get_model()

    print("Running inference benchmark...\n")
    _, load, inference, total = run_batch_inference(data_loader, net)

    print("Results...")
    print("C2.1: Data-loading times")
    print("> per epoch: ", load)
    print("> average: ", torch.mean(load))
    print("C2.2: Training time for each epoch")
    print("> per epoch", inference)
    print("> average", torch.mean(inference))
    print("C2.3: Total time for each epoch")
    print("> per epoch", total)
    print("> average", torch.mean(total))

if __name__ == "__main__":
    benchmark()