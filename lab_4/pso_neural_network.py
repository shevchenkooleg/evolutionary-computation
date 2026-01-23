import numpy as np
from typing import List, Dict, Any
from neural_network import NeuralNetwork


class PSOParticle:
    """Частица для PSO оптимизации весов нейронной сети"""

    def __init__(self, nn_architecture: Dict[str, int], bounds: tuple = (-1, 1)):
        """
        Инициализация частицы.

        Параметры:
        -----------
        nn_architecture : dict
            Архитектура нейронной сети {'input_size': 4, 'hidden_size': 8, 'output_size': 3}
        bounds : tuple
            Границы для инициализации весов
        """
        self.nn = NeuralNetwork(**nn_architecture)

        # Инициализация позиции (веса сети)
        self.position = self.initialize_position(bounds)
        self.velocity = np.zeros_like(self.position)

        # Лучшая позиция частицы
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.current_fitness = float('inf')

    def initialize_position(self, bounds: tuple) -> np.ndarray:
        """Инициализация позиции частицы (случайные веса)"""
        # Получаем текущие веса
        weights_vector = self.nn.get_weights_vector()

        # Генерируем случайные веса в заданных границах
        random_weights = np.random.uniform(
            bounds[0], bounds[1], len(weights_vector)
        )

        return random_weights

    def evaluate_fitness(self, X_train, y_train_onehot) -> float:
        """Оценка fitness частицы (меньше = лучше)"""
        # Устанавливаем веса
        self.nn.set_weights_from_vector(self.position)

        # Вычисляем loss
        loss = self.nn.get_loss(X_train, y_train_onehot)
        self.current_fitness = loss

        # Обновляем лучшую позицию
        if loss < self.best_fitness:
            self.best_fitness = loss
            self.best_position = self.position.copy()

        return loss

    def update_velocity(self, global_best_position: np.ndarray,
                        w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
                        v_max: float = 0.5):
        """Обновление скорости частицы"""
        r1 = np.random.rand(*self.velocity.shape)
        r2 = np.random.rand(*self.velocity.shape)

        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)

        self.velocity = w * self.velocity + cognitive + social

        # Ограничение скорости
        velocity_norm = np.linalg.norm(self.velocity)
        if velocity_norm > v_max:
            self.velocity = (self.velocity / velocity_norm) * v_max

    def update_position(self):
        """Обновление позиции частицы"""
        self.position = self.position + self.velocity

    def get_accuracy(self, X, y) -> float:
        """Получение точности текущей сети"""
        self.nn.set_weights_from_vector(self.position)
        return self.nn.get_accuracy(X, y)


class ModifiedPSO:
    """Модифицированный PSO для обучения нейронной сети"""

    def __init__(self, swarm_size: int = 20, nn_architecture: Dict[str, int] = None,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
                 v_max: float = 0.5, local_search_prob: float = 0.1):
        """
        Инициализация модифицированного PSO.

        Параметры:
        -----------
        swarm_size : int
            Размер роя
        nn_architecture : dict
            Архитектура нейронной сети
        w, c1, c2 : float
            Параметры PSO
        v_max : float
            Максимальная скорость
        local_search_prob : float
            Вероятность локального поиска
        """
        if nn_architecture is None:
            nn_architecture = {'input_size': 4, 'hidden_size': 8, 'output_size': 3}

        self.swarm_size = swarm_size
        self.nn_architecture = nn_architecture
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.local_search_prob = local_search_prob

        # Инициализация роя
        self.swarm = [
            PSOParticle(nn_architecture, bounds=(-1, 1))
            for _ in range(swarm_size)
        ]

        # Глобальные лучшие значения
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.global_best_accuracy = 0.0

    def run(self, X_train, y_train, y_train_onehot,
            X_val, y_val, max_iterations: int = 100,
            early_stopping_patience: int = 20):
        """
        Запуск оптимизации PSO.

        Возвращает:
        -----------
        dict : История обучения и лучшая модель
        """
        history = {
            'train_loss': [], 'val_accuracy': [],
            'global_best_loss': [], 'swarm_diversity': []
        }

        # Инициализация лучших значений
        self.initialize_global_best(X_train, y_train_onehot)

        # Переменные для early stopping
        best_val_accuracy = 0.0
        patience_counter = 0

        for iteration in range(max_iterations):
            # Оценка fitness всех частиц
            for particle in self.swarm:
                particle.evaluate_fitness(X_train, y_train_onehot)

                # Обновление глобального лучшего
                if particle.current_fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.current_fitness
                    self.global_best_position = particle.position.copy()

            # Локальный поиск (модификация)
            self.apply_local_search(X_train, y_train_onehot)

            # Обновление скоростей и позиций
            for particle in self.swarm:
                particle.update_velocity(
                    self.global_best_position,
                    self.w, self.c1, self.c2, self.v_max
                )
                particle.update_position()

            # Вычисление метрик
            train_loss = self.global_best_fitness
            val_accuracy = self.evaluate_on_validation(X_val, y_val)
            diversity = self.calculate_swarm_diversity()

            # Сохранение истории
            history['train_loss'].append(train_loss)
            history['val_accuracy'].append(val_accuracy)
            history['global_best_loss'].append(self.global_best_fitness)
            history['swarm_diversity'].append(diversity)

            # Вывод прогресса
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: "
                      f"Train Loss = {train_loss:.4f}, "
                      f"Val Accuracy = {val_accuracy:.4f}")

            # Early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at iteration {iteration}")
                break

        # Создание лучшей модели
        best_nn = NeuralNetwork(**self.nn_architecture)
        best_nn.set_weights_from_vector(self.global_best_position)
        self.global_best_accuracy = best_val_accuracy

        return {
            'best_model': best_nn,
            'best_weights': self.global_best_position,
            'best_fitness': self.global_best_fitness,
            'best_accuracy': best_val_accuracy,
            'history': history
        }

    def initialize_global_best(self, X_train, y_train_onehot):
        """Инициализация глобального лучшего решения"""
        for particle in self.swarm:
            fitness = particle.evaluate_fitness(X_train, y_train_onehot)
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()

    def apply_local_search(self, X_train, y_train_onehot):
        """Применение локального поиска к лучшим частицам"""
        # Сортируем частицы по fitness
        sorted_particles = sorted(self.swarm, key=lambda p: p.current_fitness)

        # Применяем локальный поиск к лучшим 20% частиц
        num_to_improve = max(1, int(len(self.swarm) * 0.2))

        for i in range(num_to_improve):
            if np.random.rand() < self.local_search_prob:
                self.local_search(sorted_particles[i], X_train, y_train_onehot)

    def local_search(self, particle: PSOParticle, X_train, y_train_onehot,
                     search_radius: float = 0.1, num_trials: int = 5):
        """Локальный поиск вокруг позиции частицы"""
        best_local_position = particle.position.copy()
        best_local_fitness = particle.current_fitness

        for _ in range(num_trials):
            # Генерация случайной позиции в окрестности
            trial_position = particle.position + np.random.uniform(
                -search_radius, search_radius, len(particle.position)
            )

            # Временная установка весов и оценка
            particle.nn.set_weights_from_vector(trial_position)
            trial_fitness = particle.nn.get_loss(X_train, y_train_onehot)

            if trial_fitness < best_local_fitness:
                best_local_fitness = trial_fitness
                best_local_position = trial_position.copy()

        # Обновление лучшей позиции частицы
        if best_local_fitness < particle.best_fitness:
            particle.best_fitness = best_local_fitness
            particle.best_position = best_local_position
            particle.position = best_local_position

    def evaluate_on_validation(self, X_val, y_val) -> float:
        """Оценка на валидационной выборке"""
        if self.global_best_position is None:
            return 0.0

        # Создаем временную сеть с лучшими весами
        temp_nn = NeuralNetwork(**self.nn_architecture)
        temp_nn.set_weights_from_vector(self.global_best_position)

        return temp_nn.get_accuracy(X_val, y_val)

    def calculate_swarm_diversity(self) -> float:
        """Вычисление разнообразия роя"""
        if self.global_best_position is None:
            return 0.0

        positions = np.array([p.position for p in self.swarm])
        mean_position = np.mean(positions, axis=0)

        diversity = np.mean(np.linalg.norm(positions - mean_position, axis=1))
        return diversity

    def save_history(self, filename: str = "pso_history.json"):
        """Сохранение истории обучения"""
        import json

        save_data = {
            'parameters': {
                'swarm_size': self.swarm_size,
                'w': self.w,
                'c1': self.c1,
                'c2': self.c2,
                'v_max': self.v_max,
                'local_search_prob': self.local_search_prob
            },
            'global_best': {
                'fitness': float(self.global_best_fitness),
                'accuracy': float(self.global_best_accuracy)
            }
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)