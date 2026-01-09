import torch
import torch.nn as nn

class SimpleNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1=nn.Linear(2,4)
    self.relu=nn.ReLU()
    self.layer2=nn.Linear(4,1)

  def forward(self,x):
    x=self.layer1(x)
    x=self.relu(x)
    x=self.layer2(x)
    return x


model=SimpleNN()

x = torch.tensor([[1.0, 2.0],
                  [2.0, 1.0],
                  [0.5, 0.5]])

output=model(x)
print('Output: ',output)