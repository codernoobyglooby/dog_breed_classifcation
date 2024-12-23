# -*- coding: utf-8 -*-
"""dog_classification_86%.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1eIMBFNLVgJnC8PwF8Lhxx_9Cd-9bSfhV
"""

import zipfile
import os

# Unzip the dataset
with zipfile.ZipFile('/content/archive (8).zip', 'r') as zip_ref:
    zip_ref.extractall('/content/')

# Check if the dataset is extracted correctly
#os.listdir('/content//train')

os.listdir('/content/cropped/train')

import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

batch_size=32
device='cuda' if torch.cuda.is_available() else 'cpu'

transforms=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#making a function to time the experiment
#it will track model preformance, and how fast it runs
from timeit import default_timer as timer
def print_train_time(start: float, end: float, device: torch.device=None):
  total_time=end-start
  print(f'{total_time:.3f} seconds')
  return total_time

def accuracy_fn(y_true, y_pred):
  correct=torch.eq(y_true, y_pred).sum().item()
  acc = (correct/len(y_pred))*100
  return acc

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               scheduler=None,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        if scheduler is not None:
          scheduler.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              scheduler=None,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
        if scheduler is not None:
          scheduler.step(test_loss)

train_data=datasets.ImageFolder(root='/content/cropped/train', transform=transforms)
test_data=datasets.ImageFolder(root='/content/cropped/test', transform=transforms)

train_data.classes
image, label = train_data[0]
image, label

class_names=train_data.classes
class_namesidx=train_data.class_to_idx
class_namesidx

image.shape

plt.imshow(image.permute(1,2,0))
plt.title(class_names[label])
image

train_dataloader=DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_dataloader=DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)





model=torchvision.models.efficientnet_b5(pretrained=True)

model.fc=nn.Sequential(
    nn.Linear(2048, 1024),
    nn.Dropout(p=0.3),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.Dropout(p=0.4),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.Dropout(p=0.5),
    nn.ReLU(),
    nn.Linear(256, len(train_data.classes))

)

model
model.to(device)



model=torchvision.models.efficientnet_b5(pretrained=True)
model.classifier=nn.Sequential(
    nn.Linear(2048, 1024),
    nn.Dropout(p=0.3),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.Dropout(p=0.4),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.Dropout(p=0.5),
    nn.ReLU(),
    nn.Linear(256, len(train_data.classes))

)
model

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau


loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
#scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=0.01, steps_per_epoch=len(train_data), epochs=epochs)
scheduler=ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
#scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

#training loop for CNN networks

torch.manual_seed(42)
torch.cuda.manual_seed(42)
#measure the time
from tqdm.auto import tqdm
from timeit import default_timer as timer #progress bar

#set the seed and start the timer

train_time_start_model_2=timer()

epochs=10

#training and test loop
for epoch in tqdm(range(epochs)): #rapping with tqdm is for progress bar
  print(f'Epoch: {epoch}\n')
  train_step(model=model,
             data_loader=train_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             accuracy_fn=accuracy_fn,
             scheduler=None,
             device=device)
  test_step(model=model,
            data_loader=test_dataloader,
             loss_fn=loss_fn,
             accuracy_fn=accuracy_fn,
             scheduler=scheduler,
             device=device)
train_time_end_model2=timer()
total_train_time_end_model2=print_train_time(start=train_time_start_model_2,
                                  end=train_time_end_model2, device=device)

torch.save(model,'custom-model_upd.pt')
torch.save(model,'custom-model.pth')
torch.save(model.state_dict(),'custom-model-sd_upd.pt')
torch.save(model.state_dict(),'custom-model-sd.pth')

import random
#random.seed(40)
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=16):
    test_samples.append(sample)
    test_labels.append(label)
def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device

            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)

            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())

    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)

pred_probs= make_predictions(model=model,
                             data=test_samples)

# View first two prediction probabilities list
pred_probs[:2]
pred_classes = pred_probs.argmax(dim=1)
pred_classes

sample=sample/255

sample = torch.clamp(sample, 0, 1)  # Ensure values are between 0 and 1

plt.figure(figsize=(12,12))
nrows=4
ncols=4
for i, sample in enumerate(test_samples):
  plt.subplot(nrows, ncols, i+1)

  # Plot the target image
  plt.imshow(sample.permute(1, 2, 0))

  # Find the prediction label (in text form, e.g. "Sandal")
  pred_label = class_names[pred_classes[i]]

  # Get the truth label (in text form, e.g. "T-shirt")
  truth_label = class_names[test_labels[i]]

  # Create the title text of the plot
  title_text = f"Pred: {pred_label} | Truth: {truth_label}"

  # Check for equality and change title colour accordingly
  color = "g" if pred_label == truth_label else "r"
  plt.title(title_text, fontsize=5, c=color) # green text if correct
  plt.axis(False)
plt.show()

model=torchvision.models.efficientnet_b5(pretrained=True)
model







