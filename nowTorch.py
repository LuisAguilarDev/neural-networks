# torch was not added to the dockerfile cause is a bit big library so pip install torch before running this code
import torch

x1 = torch.Tensor([2.0]).double(); x1.requires_grad = True
x2 = torch.Tensor([0.0]).double(); x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
w2 = torch.Tensor([1.0]).double(); w2.requires_grad = True
b = torch.Tensor([6.8813735870195432]).double(); b.requires_grad = True
n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

print(o.data.item())
o.backward()
print('--')
print('x2',x2.grad.item())
print('x2',w2.grad.item())
print('x2',x1.grad.item())
print('x2',w1.grad.item())