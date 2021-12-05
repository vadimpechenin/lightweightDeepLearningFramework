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

d = a+(-b)
e = (-b) + c
f = d + e

f.backward(Tensor(np.array([1, 1, 1, 1, 1])))
print(b.grad.data == np.array([-2,-2,-2,-2,-2]))

x = Tensor(np.array([[1,2,3],
                     [4,5,6]]))

x.sum(0)

x.sum(1)

print(x.expand(dim=2, copies=4))

print(x)
#1 тесты по построению нейронной сети с полученным классом тензор
#Как было раньше

np.random.seed(0)

data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([[0], [1], [0], [1]])

weights_0_1 = np.random.rand(2, 3)
weights_1_2 = np.random.rand(3, 1)

for i in range(10):
    # Predict
    layer_1 = data.dot(weights_0_1)
    layer_2 = layer_1.dot(weights_1_2)

    # Compare
    diff = (layer_2 - target)
    sqdiff = (diff * diff)
    loss = sqdiff.sum(0)  # mean squared error loss

    # Learn: this is the backpropagation piece
    layer_1_grad = diff.dot(weights_1_2.transpose())
    weight_1_2_update = layer_1.transpose().dot(diff)
    weight_0_1_update = data.transpose().dot(layer_1_grad)

    weights_1_2 -= weight_1_2_update * 0.1
    weights_0_1 -= weight_0_1_update * 0.1
    print(loss[0])

#2 Как теперь с Tensor
#np.random.seed(0)

data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

w = list()
w.append(Tensor(np.random.rand(2, 3), autograd=True))
w.append(Tensor(np.random.rand(3, 1), autograd=True))

for i in range(10):

    # Predict
    pred = data.mm(w[0]).mm(w[1])

    # Compare
    loss = ((pred - target) * (pred - target)).sum(0)

    # Learn
    loss.backward(Tensor(np.ones_like(loss.data)))

    for w_ in w:
        w_.data -= w_.grad.data * 0.1
        w_.grad.data *= 0

    print(loss)

#3 Как теперь с оптимизатором градиентного спуска
from optimization.sgd import SGD
#np.random.seed(0)

data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

w = list()
w.append(Tensor(np.random.rand(2, 3), autograd=True))
w.append(Tensor(np.random.rand(3, 1), autograd=True))

optim = SGD(parameters=w, alpha=0.1)

for i in range(10):

    # Predict
    pred = data.mm(w[0]).mm(w[1])

    # Compare
    loss = ((pred - target) * (pred - target)).sum(0)

    # Learn
    loss.backward(Tensor(np.ones_like(loss.data)))

    optim.step()

    print(loss)

#4 Как теперь с оптимизатором градиентного спуска и слоями
from layers.sequential import Sequential
from layers.linear import Linear
data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

#Архитектура сети
model = Sequential([Linear(2,3), Linear(3,1)])
w = list()
w.append(Tensor(np.random.rand(2, 3), autograd=True))
w.append(Tensor(np.random.rand(3, 1), autograd=True))

optim = SGD(parameters=model.get_parameters(), alpha=0.05)

for i in range(10):

    # Predict
    pred = model.forward(data)

    # Compare
    loss = ((pred - target) * (pred - target)).sum(0)

    # Learn
    loss.backward(Tensor(np.ones_like(loss.data)))

    optim.step()

    print(loss)
g=0

