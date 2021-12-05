from layers.layer import Layer
import numpy as np

from tensor.tensor import Tensor


class Embedding(Layer):
    #Слой векторного представления, преобразует индексы в функци активации
    def __init__(self, vocab_size, dim):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim

        # Такой способ инициализации соответствует соглашениям для алгоритма word2vec
        weight = (np.random.rand(vocab_size, dim) - 0.5) / dim

        self.weight = Tensor(weight, autograd=True)

        self.parameters.append(self.weight)

    def forward(self, input):
        return self.weight.index_select(input)