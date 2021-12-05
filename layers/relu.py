from layers.layer import Layer


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.relu()