from random import random


class Neuron:
    def __init__(self):
        self.w = [random(), random()]

    def calculate(self, x):
        return self.w[0] * x[0] + self.w[1] * x[1]

    def recalculate(self, x):
        self.w[0] += 0.5 * x[0] * x[0] * self.w[0]
        self.w[1] += 0.5 * x[1] * x[1] * self.w[1]


class NeuralNetwork:
    def __init__(self) -> None:
        self.x = [
            [0.97, 0.2],
            [1, 0],
            [-0.72, 0.7],
            [-0.67, 0.74],
            [-0.8, 0.6],
            [0, -1],
            [0.2, -0.97],
            [-0.3, -0.95]
        ]
        self.neurons = [Neuron() for _ in range(2)]

    def __str__(self):
        s = ''
        for neuron in self.neurons:
            s += str(neuron.w) + '\n'
        return s

    def start(self):
        for i in range(len(self.x)):
            for neuron in self.neurons:
                neuron.recalculate(self.x[i])

        print("Training completed")


nn = NeuralNetwork()
print(nn)
print("----------------")
nn.start()
print(nn)
