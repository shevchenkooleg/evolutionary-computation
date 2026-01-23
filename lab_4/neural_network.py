import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class NeuralNetwork:
    """Простая нейронная сеть для классификации Iris"""

    def __init__(self, input_size=4, hidden_size=8, output_size=3):
        """
        Инициализация нейронной сети.

        Параметры:
        -----------
        input_size : int
            Размер входного слоя (4 для Iris)
        hidden_size : int
            Размер скрытого слоя
        output_size : int
            Размер выходного слоя (3 для Iris)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Инициализация весов
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        """Прямое распространение"""
        # Скрытый слой
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Выходной слой
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)

        return self.a2

    def predict(self, X):
        """Предсказание классов"""
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)

    def get_accuracy(self, X, y):
        """Вычисление точности"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_loss(self, X, y_onehot):
        """Вычисление функции потерь (cross-entropy)"""
        predictions = self.forward(X)
        # Добавляем эпсилон для численной стабильности
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1. - epsilon)

        # Cross-entropy loss
        N = y_onehot.shape[0]
        loss = -np.sum(y_onehot * np.log(predictions)) / N
        return loss

    def get_weights_vector(self):
        """Получение всех весов в виде одного вектора"""
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten()
        ])

    def set_weights_from_vector(self, weights_vector):
        """Установка весов из вектора"""
        idx = 0

        # W1
        w1_size = self.input_size * self.hidden_size
        self.W1 = weights_vector[idx:idx + w1_size].reshape(
            self.input_size, self.hidden_size
        )
        idx += w1_size

        # b1
        b1_size = self.hidden_size
        self.b1 = weights_vector[idx:idx + b1_size].reshape(1, self.hidden_size)
        idx += b1_size

        # W2
        w2_size = self.hidden_size * self.output_size
        self.W2 = weights_vector[idx:idx + w2_size].reshape(
            self.hidden_size, self.output_size
        )
        idx += w2_size

        # b2
        self.b2 = weights_vector[idx:].reshape(1, self.output_size)

    @staticmethod
    def sigmoid(x):
        """Сигмоидная функция активации"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    @staticmethod
    def softmax(x):
        """Функция softmax"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def load_iris_data(test_size=0.2, random_state=42):
        """Загрузка и подготовка данных Iris"""
        iris = load_iris()
        X = iris.data
        y = iris.target

        # One-hot encoding
        encoder = OneHotEncoder(sparse_output=False)
        y_onehot = encoder.fit_transform(y.reshape(-1, 1))

        # Разделение на train/test
        X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = train_test_split(
            X, y, y_onehot, test_size=test_size, random_state=random_state, stratify=y
        )

        # Нормализация
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std

        return {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'y_train_onehot': y_train_onehot, 'y_test_onehot': y_test_onehot
        }