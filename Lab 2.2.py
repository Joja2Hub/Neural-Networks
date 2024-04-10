import numpy as np
import matplotlib.pyplot as plt

# Создание двумерного набора данных
np.random.seed(0)
X_train = np.random.rand(100, 2)  # Тренировочные данные
X_test = np.random.rand(50, 2)    # Тестовые данные

# Задаем метки классов на основе условий
y_train = np.where(X_train[:, 0] > X_train[:, 1], 1, -1)
y_test = np.where(X_test[:, 0] > X_test[:, 1], 1, -1)

# Визуализация тренировочного набора данных
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], marker='o', label='Класс 1')
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], marker='x', label='Класс -1')
plt.title('Двумерный набор данных')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

# Класс для реализации перцептрона
class Perceptron:
    def __init__(self, learning_rate=0.1, n_iter=100):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for _ in range(self.n_iter):
            for xi, yi in zip(X, y):
                update = self.learning_rate * (yi - self.predict(xi))
                self.weights += update * xi
                self.bias += update

    def predict(self, X):
        return np.where(np.dot(X, self.weights) + self.bias >= 0, 1, -1)

# Класс для реализации нейрона Адалайна
class Adaline:
    def __init__(self, learning_rate=0.01, n_iter=100):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        self.costs = []
        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.weights += self.learning_rate * X.T.dot(errors)
            self.bias += self.learning_rate * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.costs.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0, 1, -1)

# Создание и обучение моделей
perceptron = Perceptron()
perceptron.fit(X_train, y_train)

adaline = Adaline()
adaline.fit(X_train, y_train)

# Оценка точности моделей на тестовом наборе данных
perceptron_accuracy = np.mean(perceptron.predict(X_test) == y_test)
adaline_accuracy = np.mean(adaline.predict(X_test) == y_test)

print("Точность перцептрона на тестовом наборе данных:", perceptron_accuracy)
print("Точность нейрона Адалайна на тестовом наборе данных:", adaline_accuracy)