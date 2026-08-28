import torch


def local_device() -> torch.device:
    """Return the best device available in the local macOS environment."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


LOCAL_DEVICE = local_device()
