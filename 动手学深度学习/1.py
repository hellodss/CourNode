import torch

x = torch.arange(12)
print(x)

print(x.numel())

print(torch.zeros((2, 3, 4)))

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算

# 打印张量x的指数值 e的x次方
print(torch.exp(x)) 

# 广播机制
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))

print(a)
print(b)
print(a + b)  # 广播机制使得可以对不同形状的张量进行运算，自动将其扩展成相同形状





