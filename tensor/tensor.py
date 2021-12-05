"""
Класс для реализации работы с тензорами
Реализован динамический вычислительный граф при конструировании графа в процессе прямого распространения
"""

import numpy as np

class Tensor (object):

    def __init__(self, data,
                 autograd = False,
                 creators = None,
                 creation_op = None,
                 id = None):

        self.data = np.array(data)
        # Хранит операции, использовавшиеся в процессе создания данного тензора
        self.creation_op = creation_op
        #Список любых тензоров, используемых для создания текущего тензора
        self.creators = creators
        self.grad = None

        self.autograd = autograd
        self.children = {}
        if (id is None):
            self.id = np.random.randint(0, 100000)
        else:
            self.id = id

        #скорректирвоать число птомков данного тензора
        if (creators is not None):
            for c in creators:
                if (self.id not in c.children):
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    def all_children_grads_accounted_for(self):
        #Метод для проверки, получил ли тензор градиенты от всех потомков
        for id, cnt in self.children.items():
            if (cnt!=0):
                return False
        return True

    def backward(self, grad=None, grad_origin = None):
        #метод для обратного распространения

        #проверка возможности обратного распространения или ожидания градиента, в последнем случае нужно уменьшить счетчик
        if (self.autograd):

            if (grad is None):
                #Позволяет не передавать градиент единиц в первый вызов
                grad = Tensor(np.ones_like(self.data))

            if (grad_origin is not None):
                if (self.children[grad_origin.id] == 0):
                    raise Exception("cannot backprop more tan once")
                else:
                    self.children[grad_origin.id] -= 1

            #Накопление градиентов от нескольких потомков
            if (self.grad is None):
                self.grad = grad
            else:
                self.grad += grad

            # у градиентов не должно быть собственных градиентов
            assert grad.autograd == False

            if (self.creators is not None and
                    (self.all_children_grads_accounted_for() or
                     grad_origin is None)):

                if (self.creation_op == "add"):
                    # Фактическое начало обратного распространения
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)

                if (self.creation_op == "sub"):
                    self.creators[0].backward(Tensor(self.grad.data), self)
                    self.creators[1].backward(Tensor(self.grad.__neg__().data), self)

                if (self.creation_op == "mul"):
                    new = self.grad * self.creators[1]
                    self.creators[0].backward(new, self)
                    new = self.grad * self.creators[0]
                    self.creators[1].backward(new, self)

                if (self.creation_op == "mm"):
                    c0 = self.creators[0]
                    c1 = self.creators[1]
                    new = self.grad.mm(c1.transpose())
                    c0.backward(new)
                    new = self.grad.transpose().mm(c0).transpose()
                    c1.backward(new)

                if (self.creation_op == "transpose"):
                    self.creators[0].backward(self.grad.transpose())

                if ("sum" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.expand(dim,
                                                               self.creators[0].data.shape[dim]))

                if ("expand" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))

                if (self.creation_op == "neg"):
                    self.creators[0].backward(self.grad.__neg__())

                if (self.creation_op == "sigmoid"):
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (self * (ones - self)))

                if (self.creation_op == "tanh"):
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (ones - (self * self)))

                if (self.creation_op == "relu"):
                    self.creators[0].backward(self.grad * (self.grad.data > 0))

                if (self.creation_op == "index_select"):
                    new_grad = np.zeros_like(self.creators[0].data)
                    indices_ = self.index_select_indices.data.flatten()
                    grad_ = grad.data.reshape(len(indices_), -1)
                    for i in range(len(indices_)):
                        new_grad[indices_[i]] += grad_[i]
                    self.creators[0].backward(Tensor(new_grad))

                if (self.creation_op == "cross_entropy"):
                    dx = self.softmax_output - self.target_dist
                    self.creators[0].backward(Tensor(dx))

    def __add__(self, other):
        if (self.autograd and other.autograd):
            return Tensor(self.data + other.data,
                          autograd = True,
                        creators=[self, other],
                        creation_op="add")
        return Tensor(self.data + other.data)

    def __neg__(self):
        if (self.autograd):
            return Tensor(self.data * -1,
                          autograd=True,
                          creators=[self],
                          creation_op="neg")
        return Tensor(self.data * -1)

    def __sub__(self, other):
        if (self.autograd and other.autograd):
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="sub")
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if (self.autograd and other.autograd):
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="mul")
        return Tensor(self.data * other.data)

    def sum(self, dim):
        if (self.autograd):
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op="sum_" + str(dim))
        return Tensor(self.data.sum(dim))

    def expand(self, dim, copies):

        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_data = self.data.repeat(copies).reshape(list(self.data.shape) + [copies]).transpose(trans_cmd)

        if (self.autograd):
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creation_op="expand_" + str(dim))
        return Tensor(new_data)

    def transpose(self):
        if (self.autograd):
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op="transpose")

        return Tensor(self.data.transpose())

    def mm(self, x):
        if (self.autograd):
            return Tensor(self.data.dot(x.data),
                          autograd=True,
                          creators=[self, x],
                          creation_op="mm")
        return Tensor(self.data.dot(x.data))

    def sigmoid(self):
        #Функция активации сигмоидная
        if (self.autograd):
            return Tensor(1 / (1 + np.exp(-self.data)),
                          autograd=True,
                          creators=[self],
                          creation_op="sigmoid")
        return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        if (self.autograd):
            return Tensor(np.tanh(self.data),
                          autograd=True,
                          creators=[self],
                          creation_op="tanh")
        return Tensor(np.tanh(self.data))

    def relu(self):
        # Функция активации ReLU
        if (self.autograd):
            return Tensor(self.data * (self.data > 0),
                          autograd=True,
                          creators=[self],
                          creation_op="relu")
        return Tensor(self.data * (self.data > 0))

    def index_select(self, indices):
        #Поддержка индексирования,
        # гарантирующая на этапе обратного распространения размещение градиентов в тех же строках,
        #
        if (self.autograd):
            new = Tensor(self.data[indices.data],
                         autograd=True,
                         creators=[self],
                         creation_op="index_select")
            new.index_select_indices = indices
            return new
        return Tensor(self.data[indices.data])

    def cross_entropy(self, target_indices):
        #Перекрестная энтропия
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape) - 1,
                                       keepdims=True)

        t = target_indices.data.flatten()
        p = softmax_output.reshape(len(t), -1)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p) * (target_dist)).sum(1).mean()

        if (self.autograd):
            out = Tensor(loss,
                         autograd=True,
                         creators=[self],
                         creation_op="cross_entropy")
            out.softmax_output = softmax_output
            out.target_dist = target_dist
            return out

        return Tensor(loss)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())

