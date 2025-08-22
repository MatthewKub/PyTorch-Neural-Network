import torch
import os
from torch import nn # nn -- neural network 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torch import optim

# Check Device & Training Setup 
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
print("Torch Version: ", torch.__version__)
print("CUDA Available:", torch.cuda.is_available(), "\n")

# Neural Network Model 
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
        input = self.flatten(input)
        logits = self.linear_relu_stack(input)
        return logits 

# Creating and instance of NeuralNetwork and printing said instance 
model = NeuralNetwork().to(device)
print(model)


# Loading Data & Creating Storage 
training_ds = datasets.MNIST(root = 'data', train = True, download = True, transform = ToTensor())
test_ds = datasets.MNIST(root = 'data', train = False, download = True, transform = ToTensor()) 

loaders = {
    
    'train' : DataLoader(training_ds, batch_size = 100, shuffle = True, num_workers = 0),
    'test' : DataLoader(test_ds, batch_size = 100, shuffle = True, num_workers = 0),
}

epoch_losses = []
epoch_accuracies = []

# Training 
optimizer = optim.Adam(model.parameters(), lr = 0.001) #lr -- Learning Rate 
loss_function = nn.CrossEntropyLoss()

def train(epoch) :
    model.train()
    for batch_index, (data, target) in enumerate(loaders['train']):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data) # Current state of training before 
        loss = loss_function(output, target)
        loss.backward() # Back propigation 
        optimizer.step()
        if batch_index % 200 == 0 or batch_index == len(loaders["train"]) - 1 :
            print(f'Train Epoch: {epoch} [{batch_index * len(data)}/{len(loaders["train"].dataset)}  ({100. *  batch_index / len(loaders["train"]):.0f}%)]\t{loss.item():.6f}')


# Testing 
def test() :
  model.eval()
  test_loss = 0
  correct = 0

  with torch.no_grad() :
    for data, target in loaders['test'] :
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        test_loss += loss_function(output, target).item()
        pred = output.argmax(dim = 1, keepdim = True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
    test_loss /= len(loaders['test'].dataset)
    accuracy = 100. * correct / len(loaders['test'].dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy {correct}/{len(loaders["test"].dataset)} ({100. * correct / len(loaders["test"].dataset):.0f}%\n)')

    return test_loss, accuracy

for epoch in range(1, 11) :
    train(epoch)
    loss, accuracy = test()
    epoch_losses.append(loss)
    epoch_accuracies.append(accuracy)

avg_loss = sum(epoch_losses) / len(epoch_losses)
avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)
print(f"Final average loss: {avg_loss:.5f}")
print(f"Final average accuracy: {avg_accuracy:.1f}%")