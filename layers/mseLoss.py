#Класс слоя, выход функция от входов (mse)
from layers.layer import Layer


class MSELoss(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return ((pred - target)*(pred - target)).sum(0)