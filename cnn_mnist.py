import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

transform=transforms.Compose([
    transforms.ToTensor(),
])

train_dataset=datasets.MNIST(root='data',train=True,download=True,transform=transform)

test_dataset=datasets.MNIST(root='data',train=False,download=True,transform=transform)

train_loader=DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)

test_loader=DataLoader(dataset=test_dataset,batch_size=64,shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,16,kernel_size=3,padding=1)
        self.relu=nn.ReLU()
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(16,32,kernel_size=3,padding=1)
        self.fc1=nn.Linear(32*7*7,128)
        self.fc2=nn.Linear(128,10)

    def forward(self,x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.pool(x)

        x=self.conv2(x)
        x=self.relu(x)
        x=self.pool(x)
        
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)

        return x

model=CNN()

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

epochs=3

for epoch in range(epochs):
    total_loss=0
    for images,labels in train_loader:
        outputs=model(images)
        loss=criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

correct=0
total=0

with torch.no_grad():
    for images, labels in test_loader:
        outputs=model(images)
        _, predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

accuracy=100*correct/total
print(f"Test Accuracy: {accuracy:.2f}%")