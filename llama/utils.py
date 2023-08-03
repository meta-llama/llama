import torch


def get_mem_info():
    global_free_bytes, total_gpu_mem = torch.cuda.mem_get_info()
    global_free_bytes = global_free_bytes / 1024 / 1024
    total_gpu_mem = total_gpu_mem / 1024 / 1024
    return global_free_bytes, total_gpu_mem
