import numpy as np

# Функция Гаусса
def gaussian(x, c, r):
    return np.exp(-(np.linalg.norm(x - c) ** 2) / (2 * r ** 2))

# Функция для вычисления расстояния
def euclidean_distance(x, c):
    return np.linalg.norm(x - c)

# Значения независимой переменной в опытах
independent_variable = [1, 3, 5, 7, 9]

# Радиусы скрытых радиальных элементов (возьмем произвольное значение)
radius = 1

# Создание матрицы G
def create_G_matrix(inputs):
    G = np.zeros((len(inputs), 5))  # 5 скрытых нейронов
    for i, x in enumerate(inputs):
        for j, c in enumerate(independent_variable):
            G[i][j] = gaussian(x, c, radius)
    return G

# Обучающая выборка
X_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([2, 4, 6, 8, 10])

# Создание матрицы G для обучающей выборки
G = create_G_matrix(X_train)

# Рассчет весов
w = np.linalg.inv(G.T.dot(G)).dot(G.T).dot(y_train)

print("Веса w:", w)

