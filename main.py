import torch
import os
from torch import nn # nn -- neural network 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
print("Torch Version: ", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

class NeuralNetwork(nn.Module) :

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        
