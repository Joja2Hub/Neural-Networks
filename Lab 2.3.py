import numpy as np

# Определение образов для каждой буквы
X_pattern = np.array([[1, 0, 1],
                      [0, 1, 0],
                      [1, 0, 1]])

Y_pattern = np.array([[1, 0, 1],
                      [0, 1, 0],
                      [0, 1, 0]])

L_pattern = np.array([[1, 0, 0],
                      [1, 0, 0],
                      [1, 1, 1]])

I_pattern = np.array([[0, 1, 0],
                      [0, 1, 0],
                      [0, 1, 0]])

# Преобразование образов в одномерные векторы
X_flat = X_pattern.flatten()
Y_flat = Y_pattern.flatten()
L_flat = L_pattern.flatten()
I_flat = I_pattern.flatten()

# Объединение образов в обучающий набор данных
X_train = np.array([X_flat, Y_flat, L_flat, I_flat])
# Метки классов: 0 - X, 1 - Y, 2 - L, 3 - I
y_train = np.array([0, 1, 2, 3])


class NeuralNetwork:
    def __init__(self):
        # Инициализация параметров нейронной сети
        self.input_size = 9  # Размер входного слоя (9 нейронов)
        self.hidden_size = 6  # Размер скрытого слоя (6 нейронов)
        self.output_size = 4  # Размер выходного слоя (4 нейрона)

        # Инициализация весов
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        # Инициализация смещений
        self.bias_hidden = np.zeros(self.hidden_size)
        self.bias_output = np.zeros(self.output_size)

        # Параметры обучения
        self.learning_rate = 0.7
        self.num_epochs = 1000

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Прямое распространение
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output_output = self.sigmoid(self.output_input)
        return self.output_output

    def backward(self, X, y):
        # Обратное распространение
        output_error = y - self.output_output
        output_delta = output_error * self.sigmoid_derivative(self.output_output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta) * self.learning_rate

    def train(self, X, y):
        for _ in range(self.num_epochs):
            self.forward(X)
            self.backward(X, y)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


# Создание и обучение нейронной сети
nn = NeuralNetwork()
nn.train(X_train, y_train)

# Предсказание классов для тестовых данных (используем те же образы, что и для обучения)
predictions = nn.predict(X_train)
print("Предсказанные классы:", predictions)