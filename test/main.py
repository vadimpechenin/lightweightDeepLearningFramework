"""
Тестирование функционала фреймворка глубокого обучения
"""
import numpy as np

from tensor.tensor import Tensor

x = Tensor([1, 2, 3, 4, 5])
print(x)

y = Tensor([2, 2, 2, 2, 2])
print(y)

z = x + y
z.backward(Tensor(np.array([1, 1, 1, 1, 1])))
print(z.creators)
print(z.creation_op)

a = Tensor([1,2,3,4,5], autograd=True)
b = Tensor([2,2,2,2,2], autograd=True)
c = Tensor([5,4,3,2,1], autograd=True)

d = a+b
e = b + c
f = d + e

f.backward(Tensor(np.array([1, 1, 1, 1, 1])))
print(b.grad.data == np.array([2,2,2,2,2]))

#Test commit
g=0
