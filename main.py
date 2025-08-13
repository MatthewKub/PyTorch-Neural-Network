import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
print("Torch Version: ", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())