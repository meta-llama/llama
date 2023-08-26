import platform
import torch

# setting False since MPS not yet supported BFloat16 that is required for LLama2
enable_mps = False


def is_it_apple_arm():
    if platform.system() != 'Darwin':
        return False
    if platform.machine() != 'arm64':
        return False
    return True


def distrubuted_device():
    if torch.cuda.is_available():
        return "nccl"
    else:
        return "gloo"


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif is_it_apple_arm() and enable_mps:
        return torch.device("mps")
    else:
        return torch.device("cpu")

  
def model_device():
    if is_it_apple_arm() and enable_mps:
        return torch.device("mps")
    else:
        # for CUDA we also want to us CPU for model
        return torch.device("cpu")