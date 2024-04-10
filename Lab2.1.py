import numpy as np
import matplotlib.pyplot as plt

# Входные данные (координаты точек)
X = np.array([[0, 0],
              [1, 1],
              [1, 0],
              [0, 1]])

# Выходные данные (метки классов)
# Класс 0: точки (0, 0) и (1, 1)
# Класс 1: точки (1, 0) и (0, 1)
y = np.array([0, 0, 1, 1])


# Функция активации ReLU
def relu(x):
    return np.maximum(0, x)


# Инициализация весов и смещения
w = np.random.randn(2)
b = np.random.randn(1)

# Обучение нейрона с использованием градиентного спуска
learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    for i in range(len(X)):
        # Прямое распространение (предсказание)
        z = np.dot(X[i], w) + b
        a = relu(z)

        # Обратное распространение (обновление весов)
        if y[i] == 0:
            if a > 0:
                w -= learning_rate * X[i]
                b -= learning_rate
        else:
            if a <= 0:
                w += learning_rate * X[i]
                b += learning_rate

# Визуализация разделения классов
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Данные XOR')
plt.xlabel('X1')
plt.ylabel('X2')

# Построение разделяющей прямой (гиперплоскости) с учетом смещения
x_values = np.linspace(-0.5, 1.5, 100)
y_values = -(w[0] * x_values + b) / w[1]
plt.plot(x_values, y_values, color='red', linestyle='--')

plt.show()