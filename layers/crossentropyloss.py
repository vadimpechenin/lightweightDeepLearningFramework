class CrossEntropyLoss(object):
    #Класс, реализующий слой кросс-энтропии
    def __init__(self) -> object:
        super().__init__()

    def forward(self, input, target):
        return input.cross_entropy(target)