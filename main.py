import torch
import os
from torch import nn # nn -- neural network 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
print("Torch Version: ", torch.__version__)
print("CUDA Available:", torch.cuda.is_available(), "\n")

class NeuralNetwork(nn.Module) :

    def __init__(self, *args, **kwargs) :
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten() # Flattens a contiguous range of dims into a tensor. (Converts images with dimensions to a tensor/flat-vector)
        self.linear_relu_stack = nn.Sequential( # Sequential() allows the calling of the multi-layer network as a single layer 
            nn.Linear(28*28, 512), # 1st layer -- Accepts the 28*28 (784) long flatened tensor and outputs 512 features. 
            nn.ReLU(), # ReLU Activation F'n to maintain posiitve values and convert all negatives to 0.
            nn.Linear(512, 512), # 2nd layer -- Hidden layer that accepts 512 input features and outputs 512 features.
            nn.ReLU(), # # ReLU Activation F'n to maintain posiitve values and convert all negatives to 0.
            nn.Linear(512, 10), # 3rd layer -- Final fully connected layer, outputting 10 values.
        )
    
    def forward(self, input) :
        input = self.flatten()
        logits = self.linear_relu_stack(input)
        return logits 

# Creating and instance of NeuralNetwork and printing said instance 
model = NeuralNetwork().to(device)
print(model)