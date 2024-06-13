import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms

# Load the ResNet18 model pre-trained on ImageNet
model = models.resnet18(pretrained=True)

# Modify the final layer to match the number of digit classes (10)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# Set up training parameters
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Transfer learning requires freezing the convolutional layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the final fully connected layer
for param in model.fc.parameters():
    param.requires_grad = True

# Assuming `train_loader` and `val_loader` are your data loaders
# Train the model
# for epoch in range(num_epochs):
#     for inputs, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

# model.eval()
# Validation loop...
