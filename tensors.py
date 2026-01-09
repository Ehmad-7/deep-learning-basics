import torch


s=torch.tensor(5)
print('Scalar: ',s)

v=torch.tensor([1.0,2.0,3.0])
print('Vector: ',v)

m=torch.tensor([[1,2],[3,4]])
print('Matrix: ',m)

r=torch.rand(2,3)
print("Random Tensor: ",r)

print("Shape of r: ",r.shape)

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print("Add:", x + y)
print("Multiply:", x * y)