import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader

transform=transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

train_dataset=datasets.MNIST(root='data',train=False,download=True,transform=transform)

test_dataset=datasets.MNIST(root='data',train=False,download=True,transform=transform)

train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)

model=models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad=False

model.fc=nn.Linear(model.fc.in_features,10)

optimizer=optim.Adam(model.fc.parameters(),lr=0.001)
criterion=nn.CrossEntropyLoss()

epochs=2

for epoch in range(epochs):
    total_loss=0
    for images,labels in train_loader:
        outputs=model(images)
        loss=criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")


correct=0
total=0

with torch.no_grad():
    for images,labels in test_loader:
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

accuracy=100*correct/total
print(f"Test Accuracy: {accuracy:.2f}%")