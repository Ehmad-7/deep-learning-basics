import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Subset

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## with augmentation

# train_transform=transforms.Compose([
#     transforms.RandomRotation(15),
#     transforms.RandomAffine(0,translate=(0.1,0.1)),
#     transforms.ToTensor(),
# ])

## without augmentation

train_transform = transforms.Compose([
    transforms.ToTensor(),
])


test_transform=transforms.Compose([
    transforms.ToTensor(),
])

full_train_dataset=datasets.MNIST(root='data',train=True,download=True,transform=train_transform)
full_test_dataset=datasets.MNIST(root='data',train=False,download=True,transform=test_transform)

train_subset=Subset(full_train_dataset, range(5000))

train_loader=DataLoader(train_subset,batch_size=64,shuffle=True)
test_loader=DataLoader(full_test_dataset,batch_size=64,shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model=CNN().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)


epochs=5

for epoch in range(epochs):
    model.train()
    total_loss=0
    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)

        outputs=model(images)
        loss=criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")



model.eval()
correct=0
total=0

with torch.no_grad():
    for images,labels in test_loader:
        images,labels=images.to(device),labels.to(device)

        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy with Augmentation: {accuracy:.2f}%")