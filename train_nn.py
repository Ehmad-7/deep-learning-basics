import torch
import torch.nn as nn
import torch.optim as optim

X = torch.rand(100, 2)
y = X.sum(dim=1, keepdim=True)

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 8)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

model = SimpleNN()

criterion = nn.MSELoss()

# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

# training loop
epochs = 100

for epoch in range(epochs):
    # forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# test
test_input = torch.tensor([[0.2, 0.4]])
prediction = model(test_input)
print("Test Input:", test_input)
print("Prediction:", prediction.item())
print("Actual:", test_input.sum().item())
