"""
Тестирование функционала фреймворка глубокого обучения
"""
from tensor.tensor import Tensor

x = Tensor([1, 2, 3, 4, 5])
print(x)

y = x + x
print(y)
