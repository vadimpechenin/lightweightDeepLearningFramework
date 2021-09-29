"""
Класс для реализации работы с тензорами
Реализован динамический вычислительный граф при конструировании графа в процессе прямого распространения
"""

import numpy as np

class Tensor (object):

    def __init__(self, data, autograd = False, creators = None, creation_op = None, id = None):
        self.data = np.array(data)
        # Хранит операции, использовавшиеся в процессе создания данного тензора
        self.creation_op = creation_op
        #Список любых тензоров, используемых для создания текущего тензора
        self.creators = creators
        self.grad = None

        self.autograd = autograd
        self.children = {}
        if (id is None):
            id = np.random.randint(0, 100000)
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

                if (self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is None)):
                    if (self.creation_op == "add"):
                        # Фактическое начало обратного распространения
                        self.creators[0].backward(grad, self)
                        self.creators[1].backward(grad, self)

    def __add__(self, other):
        if (self.autograd and other.autograd):
            return Tensor(self.data + other.data,
                          autograd = True,
                        creators=[self, other],
                        creation_op="add")
        return Tensor(self.data + other.data)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())

