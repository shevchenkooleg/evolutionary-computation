import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import time
from tqdm import tqdm
import json
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

# ==================== КОНСТАНТЫ И ПАРАМЕТРЫ ДЛЯ ЭКСПЕРИМЕНТОВ ====================

# 1. ПАРАМЕТРЫ ФУНКЦИИ РАСТРИГИНА
RASTRIGIN_PARAMS = {
    "A": 10,  # Параметр функции
    "dimensions": [2, 5, 10, 20],  # Размерности для исследования
    "bounds": (-5.12, 5.12),  # Границы поиска
}

# 2. БАЗОВЫЕ ПАРАМЕТРЫ PSO АЛГОРИТМА
PSO_BASE_PARAMS = {
    "swarm_size": 30,  # Размер роя (количество частиц)
    "iterations": 100,  # Максимальное количество итераций
    "w": 0.7,  # Коэффициент инерции
    "c1": 1.5,  # Коэффициент когнитивного компонента (личный опыт)
    "c2": 1.5,  # Коэффициент социального компонента (коллективный опыт)
    "v_max": 1.0,  # Максимальная скорость (ограничение)
    "boundary_strategy": "reflect",  # Стратегия при выходе за границы: reflect, absorb, cyclic
}

# 3. ЗНАЧЕНИЯ ПАРАМЕТРОВ ДЛЯ ИССЛЕДОВАНИЯ
PARAM_SWEEPS = {
    "swarm_size": [10, 20, 30, 50, 100],  # Размер роя
    "w": [0.4, 0.7, 0.9, 1.2],  # Коэффициент инерции
    "c1": [0.5, 1.0, 1.5, 2.0, 2.5],  # Когнитивный коэффициент
    "c2": [0.5, 1.0, 1.5, 2.0, 2.5],  # Социальный коэффициент
    "v_max": [0.1, 0.5, 1.0, 2.0, 5.0],  # Максимальная скорость
}


# ==================== ОСНОВНЫЕ ФУНКЦИИ ====================

def rastrigin_function(x, A=10):
    """
    Вычисление функции Растригина.

    Parameters:
    -----------
    x : numpy.ndarray
        Вектор переменных
    A : float
        Параметр функции (стандартно 10)

    Returns:
    --------
    float : Значение функции Растригина
    """
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


class Particle:
    """Класс частицы для PSO."""

    def __init__(self, dim, bounds):
        """
        Инициализация частицы.

        Parameters:
        -----------
        dim : int
            Размерность задачи
        bounds : tuple
            Границы поиска (min, max)
        """
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.velocity = np.zeros(dim)
        self.best_position = self.position.copy()
        self.best_value = float('inf')
        self.value = float('inf')

    def update_velocity(self, gbest_position, w, c1, c2, v_max):
        """
        Обновление скорости частицы.

        Parameters:
        -----------
        gbest_position : numpy.ndarray
            Глобально лучшее положение
        w : float
            Коэффициент инерции
        c1 : float
            Когнитивный коэффициент
        c2 : float
            Социальный коэффициент
        v_max : float
            Максимальная скорость
        """
        r1, r2 = np.random.rand(2)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (gbest_position - self.position)

        self.velocity = w * self.velocity + cognitive + social

        # Ограничение скорости
        velocity_norm = np.linalg.norm(self.velocity)
        if velocity_norm > v_max:
            self.velocity = (self.velocity / velocity_norm) * v_max

    def update_position(self, bounds, boundary_strategy="reflect"):
        """
        Обновление позиции частицы.

        Parameters:
        -----------
        bounds : tuple
            Границы поиска
        boundary_strategy : str
            Стратегия обработки границ
        """
        new_position = self.position + self.velocity

        # Обработка границ
        if boundary_strategy == "reflect":
            # Отражение от границ
            for i in range(len(new_position)):
                if new_position[i] < bounds[0]:
                    new_position[i] = 2 * bounds[0] - new_position[i]
                    self.velocity[i] = -self.velocity[i]
                elif new_position[i] > bounds[1]:
                    new_position[i] = 2 * bounds[1] - new_position[i]
                    self.velocity[i] = -self.velocity[i]

        elif boundary_strategy == "absorb":
            # Поглощение (остановка на границе)
            new_position = np.clip(new_position, bounds[0], bounds[1])

        elif boundary_strategy == "cyclic":
            # Циклическое возвращение (тор)
            for i in range(len(new_position)):
                if new_position[i] < bounds[0]:
                    new_position[i] = bounds[1] - (bounds[0] - new_position[i])
                elif new_position[i] > bounds[1]:
                    new_position[i] = bounds[0] + (new_position[i] - bounds[1])

        self.position = new_position


def initialize_swarm(swarm_size, dim, bounds):
    """
    Инициализация роя частиц.

    Parameters:
    -----------
    swarm_size : int
        Размер роя
    dim : int
        Размерность задачи
    bounds : tuple
        Границы поиска

    Returns:
    --------
    list : Список объектов Particle
    """
    return [Particle(dim, bounds) for _ in range(swarm_size)]


def pso_algorithm(n_dim=2, **pso_params):
    """
    Основная функция алгоритма оптимизации роем частиц.

    Parameters:
    -----------
    n_dim : int
        Размерность задачи
    pso_params : dict
        Параметры PSO

    Returns:
    --------
    dict : Результаты оптимизации
    """
    # Извлечение параметров
    swarm_size = pso_params.get("swarm_size", 30)
    iterations = pso_params.get("iterations", 100)
    w = pso_params.get("w", 0.7)
    c1 = pso_params.get("c1", 1.5)
    c2 = pso_params.get("c2", 1.5)
    v_max = pso_params.get("v_max", 1.0)
    boundary_strategy = pso_params.get("boundary_strategy", "reflect")
    A = pso_params.get("A", 10)
    bounds = pso_params.get("bounds", (-5.12, 5.12))

    # Параметры остановки
    max_iterations = pso_params.get("max_iterations", 100)
    target_value = pso_params.get("target_value", 1e-4)
    stagnation_iter = pso_params.get("stagnation_iterations", 20)

    # Инициализация
    start_time = time.time()
    swarm = initialize_swarm(swarm_size, n_dim, bounds)

    # Инициализация лучших значений
    gbest_position = None
    gbest_value = float('inf')
    gbest_particle_idx = 0

    # История
    history = {
        'best_value': [],  # Лучшее значение функции
        'avg_value': [],  # Среднее значение функции
        'best_position': [],  # Лучшая позиция
        'swarm_diversity': [],  # Разнообразие роя
        'iteration_time': [],  # Время на итерацию
        'velocity_norms': [],  # Нормы скоростей частиц
        'positions_history': [],  # История позиций (только для 2D)
    }

    # Переменные для отслеживания стагнации
    stagnation_counter = 0
    converged = False

    # Основной цикл PSO
    for iter_num in range(max_iterations):
        iter_start_time = time.time()

        # 1. Оценка текущих позиций
        for i, particle in enumerate(swarm):
            particle.value = rastrigin_function(particle.position, A)

            # Обновление личного лучшего
            if particle.value < particle.best_value:
                particle.best_value = particle.value
                particle.best_position = particle.position.copy()

            # Обновление глобального лучшего
            if particle.value < gbest_value:
                gbest_value = particle.value
                gbest_position = particle.position.copy()
                gbest_particle_idx = i

        # 2. Сохранение статистики
        values = [p.value for p in swarm]
        best_val = gbest_value
        avg_val = np.mean(values)

        # Вычисление разнообразия роя (среднее расстояние до gbest)
        if swarm_size > 1:
            distances = [np.linalg.norm(p.position - gbest_position) for p in swarm]
            diversity = np.mean(distances)
        else:
            diversity = 0

        # Вычисление средних норм скоростей
        velocity_norms = [np.linalg.norm(p.velocity) for p in swarm]
        avg_velocity = np.mean(velocity_norms)

        # Сохранение в историю
        history['best_value'].append(best_val)
        history['avg_value'].append(avg_val)
        history['best_position'].append(gbest_position.copy())
        history['swarm_diversity'].append(diversity)
        history['velocity_norms'].append(avg_velocity)

        # Для 2D сохраняем позиции для визуализации
        if n_dim == 2 and iter_num % 10 == 0:  # Каждые 10 итераций
            positions = np.array([p.position for p in swarm])
            history['positions_history'].append({
                'iteration': iter_num,
                'positions': positions.copy(),
                'gbest': gbest_position.copy()
            })

        # 3. Проверка критериев остановки
        # а) Достигнута целевая точность
        if best_val < target_value:
            converged = True
            break

        # б) Проверка стагнации (улучшения нет в течение N итераций)
        if iter_num > 0:
            improvement = history['best_value'][-2] - history['best_value'][-1]
            if improvement < 1e-6:  # Улучшение меньше порога
                stagnation_counter += 1
            else:
                stagnation_counter = 0

        if stagnation_counter >= stagnation_iter:
            converged = False  # Стагнация
            break

        # 4. Обновление скоростей и позиций частиц
        for particle in swarm:
            particle.update_velocity(gbest_position, w, c1, c2, v_max)
            particle.update_position(bounds, boundary_strategy)

        # Время итерации
        history['iteration_time'].append(time.time() - iter_start_time)

    # Формирование результатов
    total_time = time.time() - start_time

    result = {
        'best_position': gbest_position,
        'best_value': gbest_value,
        'converged': converged,
        'iterations': iter_num + 1,
        'total_time': total_time,
        'history': history,
        'parameters': {
            'n_dim': n_dim,
            'swarm_size': swarm_size,
            'w': w,
            'c1': c1,
            'c2': c2,
            'v_max': v_max,
            'boundary_strategy': boundary_strategy,
        }
    }

    return result


# ==================== ВИЗУАЛИЗАЦИЯ ====================

def visualize_rastrigin_2d(A=10, bounds=(-5.12, 5.12)):
    """Визуализация функции Растригина для 2D случая."""
    # Создание сетки
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)

    # Вычисление значений функции
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = rastrigin_function(point, A)

    # Создание графиков
    fig = plt.figure(figsize=(14, 6))

    # 3D поверхность
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis,
                            alpha=0.8, linewidth=0, antialiased=True)
    ax1.set_xlabel('x₁', fontsize=12)
    ax1.set_ylabel('x₂', fontsize=12)
    ax1.set_zlabel('f(x₁, x₂)', fontsize=12)
    ax1.set_title('Функция Растригина (3D поверхность)', fontsize=14, fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

    # Линии уровня
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contour(X, Y, Z, levels=30, cmap=cm.viridis)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_xlabel('x₁', fontsize=12)
    ax2.set_ylabel('x₂', fontsize=12)
    ax2.set_title('Линии уровня функции Растригина', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Отметка глобального минимума
    ax2.plot(0, 0, 'r*', markersize=15, label='Глобальный минимум (0, 0)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('rastrigin_function_pso.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def plot_convergence_curves_pso(results_dict, title="Сравнение сходимости PSO"):
    """Построение графиков сходимости для разных конфигураций PSO."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    colors = plt.cm.rainbow(np.linspace(0, 1, len(results_dict)))

    for (label, result), color in zip(results_dict.items(), colors):
        history = result['history']
        if len(history['best_value']) == 0:
            continue

        iterations = list(range(1, len(history['best_value']) + 1))

        # График 1: Лучшее значение функции
        axes[0, 0].semilogy(iterations, history['best_value'],
                            label=label, color=color, linewidth=2)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Итерация', fontsize=12)
        axes[0, 0].set_ylabel('Лучшее f(x) (log scale)', fontsize=12)
        axes[0, 0].set_title('Сходимость лучшего значения', fontsize=13)
        axes[0, 0].grid(True, alpha=0.3)

        # График 2: Среднее значение функции
        axes[0, 1].plot(iterations, history['avg_value'],
                        label=label, color=color, linewidth=2)
        axes[0, 1].set_xlabel('Итерация', fontsize=12)
        axes[0, 1].set_ylabel('Среднее f(x)', fontsize=12)
        axes[0, 1].set_title('Эволюция среднего значения', fontsize=13)
        axes[0, 1].grid(True, alpha=0.3)

        # График 3: Разнообразие роя
        axes[1, 0].plot(iterations, history['swarm_diversity'],
                        label=label, color=color, linewidth=2)
        axes[1, 0].set_xlabel('Итерация', fontsize=12)
        axes[1, 0].set_ylabel('Разнообразие роя', fontsize=12)
        axes[1, 0].set_title('Динамика разнообразия', fontsize=13)
        axes[1, 0].grid(True, alpha=0.3)

        # График 4: Скорости частиц
        axes[1, 1].plot(iterations, history['velocity_norms'],
                        label=label, color=color, linewidth=2)
        axes[1, 1].set_xlabel('Итерация', fontsize=12)
        axes[1, 1].set_ylabel('Средняя норма скорости', fontsize=12)
        axes[1, 1].set_title('Динамика скоростей частиц', fontsize=13)
        axes[1, 1].grid(True, alpha=0.3)

    # Добавляем легенду
    axes[0, 0].legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    plt.savefig('pso_convergence_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def plot_particle_trajectories(result, bounds=(-5.12, 5.12)):
    """
    Визуализация траекторий частиц в 2D пространстве.
    Работает только для 2D случая.
    """
    if result['parameters']['n_dim'] != 2:
        print("Визуализация траекторий возможна только для 2D!")
        return None

    # Подготовка данных
    positions_history = result['history']['positions_history']
    if not positions_history:
        print("История позиций не сохранена!")
        return None

    # Создание сетки для функции
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = rastrigin_function(point)

    # Создание анимации или нескольких кадров
    n_frames = min(5, len(positions_history))
    step = max(1, len(positions_history) // n_frames)

    fig, axes = plt.subplots(1, n_frames, figsize=(5 * n_frames, 4))
    if n_frames == 1:
        axes = [axes]

    for idx, frame_idx in enumerate(range(0, len(positions_history), step)):
        if idx >= n_frames:
            break

        data = positions_history[frame_idx]
        positions = data['positions']
        gbest = data['gbest']

        # Линии уровня
        axes[idx].contour(X, Y, Z, levels=20, cmap=cm.Greys, alpha=0.5)

        # Точки - позиции частиц
        axes[idx].scatter(positions[:, 0], positions[:, 1],
                          c='blue', s=30, alpha=0.6, label='Частицы')

        # Лучшая позиция
        axes[idx].scatter(gbest[0], gbest[1],
                          c='red', s=100, marker='*', label='Лучшая позиция')

        # Глобальный минимум
        axes[idx].scatter(0, 0, c='green', s=100, marker='X', label='Глобальный минимум')

        axes[idx].set_xlim(bounds[0], bounds[1])
        axes[idx].set_ylim(bounds[0], bounds[1])
        axes[idx].set_xlabel('x₁')
        axes[idx].set_ylabel('x₂')
        axes[idx].set_title(f'Итерация {data["iteration"]}')
        axes[idx].grid(True, alpha=0.3)

        if idx == 0:
            axes[idx].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('pso_particle_trajectories.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def plot_parameter_sweep_results_pso(sweep_results, param_name, title=None):
    """Визуализация результатов сканирования параметра PSO."""
    if title is None:
        title = f'Влияние параметра {param_name} на эффективность PSO'

    param_values = list(sweep_results.keys())
    best_values = [r['best_value'] for r in sweep_results.values()]

    # Для вычисления средних значений
    avg_values = []
    for val in param_values:
        if 'avg_value' in sweep_results[val]:
            avg_values.append(sweep_results[val]['avg_value'])
        else:
            # Вычисляем среднее из истории
            if 'all_runs' in sweep_results[val] and len(sweep_results[val]['all_runs']) > 0:
                history_avg = np.mean(sweep_results[val]['all_runs'][0]['history']['avg_value'])
            else:
                history_avg = sweep_results[val]['best_value']
            avg_values.append(history_avg)

    converged = [r.get('converged', True) for r in sweep_results.values()]
    iterations = [r.get('iterations', 100) for r in sweep_results.values()]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # График 1: Лучшее значение vs параметр
    ax1 = axes[0, 0]

    # Создаем отдельные scatter-объекты для разных типов точек
    from matplotlib.lines import Line2D

    for val, best, conv in zip(param_values, best_values, converged):
        if conv:
            ax1.scatter(val, best, marker='o', s=100, color='green', alpha=0.7)
        else:
            ax1.scatter(val, best, marker='x', s=100, color='red', alpha=0.7)

    # Соединяем все точки линией
    ax1.plot(param_values, best_values, 'b-', alpha=0.3)
    ax1.set_xlabel(param_name, fontsize=12)
    ax1.set_ylabel('Лучшее f(x)', fontsize=12)
    ax1.set_title('Качество решения', fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Создаем кастомную легенду
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=10, label='Сошелся', alpha=0.7),
        Line2D([0], [0], marker='x', color='red', markerfacecolor='red',
               markersize=10, label='Не сошелся', alpha=0.7, markeredgewidth=2)
    ]
    ax1.legend(handles=legend_elements, loc='best')

    # График 2: Число итераций vs параметр
    ax2 = axes[0, 1]
    ax2.plot(param_values, iterations, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel(param_name, fontsize=12)
    ax2.set_ylabel('Число итераций', fontsize=12)
    ax2.set_title('Скорость сходимости', fontsize=13)
    ax2.grid(True, alpha=0.3)

    # График 3: Среднее значение vs параметр
    ax3 = axes[1, 0]
    ax3.plot(param_values, avg_values, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel(param_name, fontsize=12)
    ax3.set_ylabel('Среднее f(x)', fontsize=12)
    ax3.set_title('Среднее качество роя', fontsize=13)
    ax3.grid(True, alpha=0.3)

    # График 4: Гистограмма успешности
    ax4 = axes[1, 1]
    success_count = sum(converged)
    total_count = len(converged)
    success_rate = success_count / total_count * 100 if total_count > 0 else 0

    bars = ax4.bar(['Сошелся', 'Не сошелся'],
                   [success_rate, 100 - success_rate],
                   color=['green', 'red'], alpha=0.7)
    ax4.set_ylabel('Процент случаев (%)', fontsize=12)
    ax4.set_title(f'Успешность: {success_rate:.1f}%', fontsize=13)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Формируем имя файла
    safe_param_name = param_name.replace(' ', '_').replace('(', '').replace(')', '')
    filename = f'pso_{safe_param_name}_experiment.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    return fig


# ==================== ЭКСПЕРИМЕНТЫ ====================

def experiment_swarm_size(n_dim=2, n_runs=3):
    """Эксперимент: влияние размера роя."""
    print(f"\n{'=' * 60}")
    print("ЭКСПЕРИМЕНТ: Влияние размера роя")
    print(f"{'=' * 60}")

    results = {}
    param_values = PARAM_SWEEPS["swarm_size"]

    for swarm_size in tqdm(param_values, desc="Размер роя"):
        run_results = []

        for run in range(n_runs):
            pso_params = PSO_BASE_PARAMS.copy()
            pso_params["swarm_size"] = swarm_size

            result = pso_algorithm(n_dim=n_dim, **pso_params)
            run_results.append(result)

        # Агрегируем результаты по запускам
        avg_best = np.mean([r['best_value'] for r in run_results])
        avg_iterations = np.mean([r['iterations'] for r in run_results])
        success_rate = np.mean([r['converged'] for r in run_results])

        results[swarm_size] = {
            'best_value': avg_best,
            'avg_value': np.mean([np.mean(r['history']['avg_value']) for r in run_results]),
            'iterations': avg_iterations,
            'converged': success_rate > 0.5,
            'success_rate': success_rate,
            'all_runs': run_results
        }

        print(f"S={swarm_size:3d}: f={avg_best:.6f}, iter={avg_iterations:.1f}, "
              f"success={success_rate * 100:.0f}%")

    # Визуализация
    fig = plot_parameter_sweep_results_pso(results, "swarm_size",
                                           "Влияние размера роя на эффективность PSO")
    return results, fig


def experiment_inertia_weight(n_dim=2, n_runs=3):
    """Эксперимент: влияние коэффициента инерции w."""
    print(f"\n{'=' * 60}")
    print("ЭКСПЕРИМЕНТ: Влияние коэффициента инерции w")
    print(f"{'=' * 60}")

    results = {}
    param_values = PARAM_SWEEPS["w"]

    for w in tqdm(param_values, desc="Коэффициент инерции"):
        run_results = []

        for run in range(n_runs):
            pso_params = PSO_BASE_PARAMS.copy()
            pso_params["w"] = w

            result = pso_algorithm(n_dim=n_dim, **pso_params)
            run_results.append(result)

        avg_best = np.mean([r['best_value'] for r in run_results])
        avg_iterations = np.mean([r['iterations'] for r in run_results])
        success_rate = np.mean([r['converged'] for r in run_results])

        results[w] = {
            'best_value': avg_best,
            'avg_value': np.mean([np.mean(r['history']['avg_value']) for r in run_results]),
            'iterations': avg_iterations,
            'converged': success_rate > 0.5,
            'success_rate': success_rate,
            'all_runs': run_results
        }

        print(f"w={w:.2f}: f={avg_best:.6f}, iter={avg_iterations:.1f}, "
              f"success={success_rate * 100:.0f}%")

    # Визуализация
    fig = plot_parameter_sweep_results_pso(results, "inertia_weight",
                                           "Влияние коэффициента инерции на эффективность PSO")
    return results, fig


def experiment_cognitive_coefficient(n_dim=2, n_runs=3):
    """Эксперимент: влияние когнитивного коэффициента c1."""
    print(f"\n{'=' * 60}")
    print("ЭКСПЕРИМЕНТ: Влияние когнитивного коэффициента c1")
    print(f"{'=' * 60}")

    results = {}
    param_values = PARAM_SWEEPS["c1"]

    for c1 in tqdm(param_values, desc="Когнитивный коэффициент"):
        run_results = []

        for run in range(n_runs):
            pso_params = PSO_BASE_PARAMS.copy()
            pso_params["c1"] = c1

            result = pso_algorithm(n_dim=n_dim, **pso_params)
            run_results.append(result)

        avg_best = np.mean([r['best_value'] for r in run_results])
        avg_iterations = np.mean([r['iterations'] for r in run_results])
        success_rate = np.mean([r['converged'] for r in run_results])

        results[c1] = {
            'best_value': avg_best,
            'avg_value': np.mean([np.mean(r['history']['avg_value']) for r in run_results]),
            'iterations': avg_iterations,
            'converged': success_rate > 0.5,
            'success_rate': success_rate,
            'all_runs': run_results
        }

        print(f"c1={c1:.2f}: f={avg_best:.6f}, iter={avg_iterations:.1f}, "
              f"success={success_rate * 100:.0f}%")

    # Визуализация
    fig = plot_parameter_sweep_results_pso(results, "cognitive_coefficient",
                                           "Влияние когнитивного коэффициента на эффективность PSO")
    return results, fig


def experiment_social_coefficient(n_dim=2, n_runs=3):
    """Эксперимент: влияние социального коэффициента c2."""
    print(f"\n{'=' * 60}")
    print("ЭКСПЕРИМЕНТ: Влияние социального коэффициента c2")
    print(f"{'=' * 60}")

    results = {}
    param_values = PARAM_SWEEPS["c2"]

    for c2 in tqdm(param_values, desc="Социальный коэффициент"):
        run_results = []

        for run in range(n_runs):
            pso_params = PSO_BASE_PARAMS.copy()
            pso_params["c2"] = c2

            result = pso_algorithm(n_dim=n_dim, **pso_params)
            run_results.append(result)

        avg_best = np.mean([r['best_value'] for r in run_results])
        avg_iterations = np.mean([r['iterations'] for r in run_results])
        success_rate = np.mean([r['converged'] for r in run_results])

        results[c2] = {
            'best_value': avg_best,
            'avg_value': np.mean([np.mean(r['history']['avg_value']) for r in run_results]),
            'iterations': avg_iterations,
            'converged': success_rate > 0.5,
            'success_rate': success_rate,
            'all_runs': run_results
        }

        print(f"c2={c2:.2f}: f={avg_best:.6f}, iter={avg_iterations:.1f}, "
              f"success={success_rate * 100:.0f}%")

    # Визуализация
    fig = plot_parameter_sweep_results_pso(results, "social_coefficient",
                                           "Влияние социального коэффициента на эффективность PSO")
    return results, fig


def experiment_vmax(n_dim=2, n_runs=3):
    """Эксперимент: влияние максимальной скорости v_max."""
    print(f"\n{'=' * 60}")
    print("ЭКСПЕРИМЕНТ: Влияние максимальной скорости v_max")
    print(f"{'=' * 60}")

    results = {}
    param_values = PARAM_SWEEPS["v_max"]

    for v_max in tqdm(param_values, desc="Максимальная скорость"):
        run_results = []

        for run in range(n_runs):
            pso_params = PSO_BASE_PARAMS.copy()
            pso_params["v_max"] = v_max

            result = pso_algorithm(n_dim=n_dim, **pso_params)
            run_results.append(result)

        avg_best = np.mean([r['best_value'] for r in run_results])
        avg_iterations = np.mean([r['iterations'] for r in run_results])
        success_rate = np.mean([r['converged'] for r in run_results])

        results[v_max] = {
            'best_value': avg_best,
            'avg_value': np.mean([np.mean(r['history']['avg_value']) for r in run_results]),
            'iterations': avg_iterations,
            'converged': success_rate > 0.5,
            'success_rate': success_rate,
            'all_runs': run_results
        }

        print(f"v_max={v_max:.2f}: f={avg_best:.6f}, iter={avg_iterations:.1f}, "
              f"success={success_rate * 100:.0f}%")

    # Визуализация
    fig = plot_parameter_sweep_results_pso(results, "v_max",
                                           "Влияние максимальной скорости на эффективность PSO")
    return results, fig


def experiment_dimensions_pso(n_runs=3):
    """Эксперимент: влияние размерности задачи на PSO."""
    print(f"\n{'=' * 60}")
    print("ЭКСПЕРИМЕНТ: Влияние размерности задачи на PSO")
    print(f"{'=' * 60}")

    results = {}
    dimensions = RASTRIGIN_PARAMS["dimensions"]

    for n_dim in tqdm(dimensions, desc="Размерность"):
        run_results = []

        for run in range(n_runs):
            # Корректируем параметры для большей размерности
            pso_params = PSO_BASE_PARAMS.copy()
            if n_dim > 5:
                # Увеличиваем размер роя для больших размерностей
                pso_params["swarm_size"] = 50
                pso_params["iterations"] = 200

            result = pso_algorithm(n_dim=n_dim, **pso_params)
            run_results.append(result)

        avg_best = np.mean([r['best_value'] for r in run_results])
        avg_iterations = np.mean([r['iterations'] for r in run_results])
        success_rate = np.mean([r['converged'] for r in run_results])

        results[n_dim] = {
            'best_value': avg_best,
            'avg_value': np.mean([np.mean(r['history']['avg_value']) for r in run_results]),
            'iterations': avg_iterations,
            'converged': success_rate > 0.5,
            'success_rate': success_rate,
            'all_runs': run_results
        }

        print(f"n={n_dim:2d}: f={avg_best:.6f}, iter={avg_iterations:.1f}, "
              f"success={success_rate * 100:.0f}%")

    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Влияние размерности задачи на эффективность PSO',
                 fontsize=16, fontweight='bold')

    dimensions_list = list(results.keys())
    best_values = [results[d]['best_value'] for d in dimensions_list]
    avg_values = [results[d]['avg_value'] for d in dimensions_list]
    iterations = [results[d]['iterations'] for d in dimensions_list]
    success_rates = [results[d]['success_rate'] for d in dimensions_list]

    # График 1: Качество решения
    axes[0, 0].plot(dimensions_list, best_values, 'ro-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Размерность (n)', fontsize=12)
    axes[0, 0].set_ylabel('Лучшее f(x)', fontsize=12)
    axes[0, 0].set_title('Качество решения', fontsize=13)
    axes[0, 0].grid(True, alpha=0.3)
    if max(best_values) > 0:
        axes[0, 0].set_yscale('log')

    # График 2: Требуемое число итераций
    axes[0, 1].plot(dimensions_list, iterations, 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Размерность (n)', fontsize=12)
    axes[0, 1].set_ylabel('Число итераций', fontsize=12)
    axes[0, 1].set_title('Скорость сходимости', fontsize=13)
    axes[0, 1].grid(True, alpha=0.3)

    # График 3: Успешность
    axes[1, 0].bar([str(d) for d in dimensions_list],
                   [r * 100 for r in success_rates],
                   color=['green' if r > 0.5 else 'red' for r in success_rates],
                   alpha=0.7)
    axes[1, 0].set_xlabel('Размерность (n)', fontsize=12)
    axes[1, 0].set_ylabel('Успешность (%)', fontsize=12)
    axes[1, 0].set_title('Процент успешных запусков', fontsize=13)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # График 4: Среднее качество роя
    axes[1, 1].plot(dimensions_list, avg_values, 'bo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Размерность (n)', fontsize=12)
    axes[1, 1].set_ylabel('Среднее f(x)', fontsize=12)
    axes[1, 1].set_title('Среднее качество роя', fontsize=13)
    axes[1, 1].grid(True, alpha=0.3)
    if max(avg_values) > 0:
        axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig('pso_dimension_experiment.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results, fig


def experiment_compare_c1_c2(n_dim=2, n_runs=3):
    """Эксперимент: сравнение баланса c1 и c2."""
    print(f"\n{'=' * 60}")
    print("ЭКСПЕРИМЕНТ: Сравнение баланса c1 и c2")
    print(f"{'=' * 60}")

    configs = {
        'Личный опыт (c1=2.5, c2=0.5)': {'c1': 2.5, 'c2': 0.5},
        'Коллективный опыт (c1=0.5, c2=2.5)': {'c1': 0.5, 'c2': 2.5},
        'Баланс (c1=1.5, c2=1.5)': {'c1': 1.5, 'c2': 1.5},
        'Активный поиск (c1=2.0, c2=2.0)': {'c1': 2.0, 'c2': 2.0},
    }

    results = {}

    for name, params in tqdm(configs.items(), desc="Конфигурации c1/c2"):
        run_results = []

        for run in range(n_runs):
            pso_params = PSO_BASE_PARAMS.copy()
            pso_params.update(params)

            result = pso_algorithm(n_dim=n_dim, **pso_params)
            run_results.append(result)

        avg_best = np.mean([r['best_value'] for r in run_results])
        avg_iterations = np.mean([r['iterations'] for r in run_results])
        success_rate = np.mean([r['converged'] for r in run_results])

        results[name] = {
            'best_value': avg_best,
            'avg_value': np.mean([np.mean(r['history']['avg_value']) for r in run_results]),
            'iterations': avg_iterations,
            'converged': success_rate > 0.5,
            'success_rate': success_rate,
            'all_runs': run_results
        }

        print(f"{name:30s}: f={avg_best:.6f}, iter={avg_iterations:.1f}, "
              f"success={success_rate * 100:.0f}%")

    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Сравнение баланса когнитивного и социального компонентов',
                 fontsize=16, fontweight='bold')

    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))

    for (name, result), color in zip(results.items(), colors):
        history = result['all_runs'][0]['history']
        iterations = list(range(1, len(history['best_value']) + 1))

        # График сходимости
        axes[0, 0].semilogy(iterations, history['best_value'],
                            label=name, color=color, linewidth=2)

        # График разнообразия
        axes[0, 1].plot(iterations, history['swarm_diversity'],
                        label=name, color=color, linewidth=2)

        # График скоростей
        axes[1, 0].plot(iterations, history['velocity_norms'],
                        label=name, color=color, linewidth=2)

        # График средних значений
        axes[1, 1].plot(iterations, history['avg_value'],
                        label=name, color=color, linewidth=2)

    # Настройка осей
    axes[0, 0].set_xlabel('Итерация')
    axes[0, 0].set_ylabel('Лучшее f(x)')
    axes[0, 0].set_title('Сходимость')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_xlabel('Итерация')
    axes[0, 1].set_ylabel('Разнообразие')
    axes[0, 1].set_title('Разнообразие роя')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Итерация')
    axes[1, 0].set_ylabel('Норма скорости')
    axes[1, 0].set_title('Скорости частиц')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Итерация')
    axes[1, 1].set_ylabel('Среднее f(x)')
    axes[1, 1].set_title('Среднее качество')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pso_c1_c2_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results, fig


# ==================== СОХРАНЕНИЕ И ЗАГРУЗКА РЕЗУЛЬТАТОВ ====================

def save_results_to_file(results, filename='pso_experiment_results.json'):
    """Сохранение результатов экспериментов в JSON файл."""

    def convert_for_json(obj):
        """Рекурсивно конвертирует объекты для JSON."""
        if isinstance(obj, (int, float, str, bool)):
            return obj
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            # Преобразуем ключи в строки
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(item) for item in obj]
        elif obj is None:
            return None
        else:
            # Для других типов пытаемся преобразовать в строку
            try:
                return str(obj)
            except:
                return f"<{type(obj).__name__}>"

    try:
        # Конвертируем все результаты
        json_ready_results = convert_for_json(results)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_ready_results, f, indent=2)

        print(f"\n✅ Результаты сохранены в файл: {filename}")
        return filename
    except Exception as e:
        print(f"\n❌ Ошибка при сохранении результатов: {e}")
        print("Попытка сохранить упрощенные результаты...")

        try:
            # Пробуем сохранить только основные данные
            simple_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    simple_value = {}
                    for subkey, subvalue in value.items():
                        if subkey in ['best_value', 'iterations', 'total_time', 'success_rate', 'avg_value']:
                            if isinstance(subvalue, (int, float)):
                                simple_value[subkey] = subvalue
                            elif isinstance(subvalue, np.number):
                                simple_value[subkey] = float(subvalue)
                        elif subkey == 'converged':
                            simple_value[subkey] = bool(subvalue)
                        elif subkey == 'best_position' and isinstance(subvalue, np.ndarray):
                            simple_value[subkey] = subvalue.tolist()
                        elif subkey == 'parameters' and isinstance(subvalue, dict):
                            # Сохраняем только основные параметры
                            simple_params = {}
                            for param_key, param_value in subvalue.items():
                                if isinstance(param_value, (int, float, str, bool)):
                                    simple_params[param_key] = param_value
                                elif isinstance(param_value, np.number):
                                    simple_params[param_key] = float(param_value)
                            simple_value[subkey] = simple_params
                    simple_results[key] = simple_value
                else:
                    simple_results[key] = str(value)[:100]  # Ограничиваем длину

            with open(f'simple_{filename}', 'w', encoding='utf-8') as f:
                json.dump(simple_results, f, indent=2)

            print(f"✅ Упрощенные результаты сохранены в файл: simple_{filename}")
            return f'simple_{filename}'
        except Exception as e2:
            print(f"❌ Не удалось сохранить даже упрощенные результаты: {e2}")
            return None


def load_results_from_file(filename='pso_experiment_results.json'):
    """Загрузка результатов экспериментов из JSON файла."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"\n✅ Результаты загружены из файла: {filename}")
        return results
    except FileNotFoundError:
        print(f"\n❌ Файл {filename} не найден!")
        # Пробуем загрузить упрощенную версию
        try:
            with open(f'simple_{filename}', 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"✅ Упрощенные результаты загружены из файла: simple_{filename}")
            return results
        except FileNotFoundError:
            print(f"❌ Файл simple_{filename} также не найден!")
            return None
    except Exception as e:
        print(f"\n❌ Ошибка при загрузке результатов: {e}")
        return None


def generate_report(results, filename='pso_lab_report.txt'):
    """Генерация текстового отчета по результатам экспериментов."""
    report_lines = []

    report_lines.append("=" * 70)
    report_lines.append("ОТЧЕТ ПО ЛАБОРАТОРНОЙ РАБОТЕ: АЛГОРИТМ PSO")
    report_lines.append("=" * 70)
    report_lines.append(f"Дата генерации: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Добавляем информацию о базовом запуске
    if 'base_run' in results:
        base = results['base_run']
        report_lines.append("1. БАЗОВЫЙ ЗАПУСК PSO")
        report_lines.append("-" * 40)
        if isinstance(base, dict):
            if 'parameters' in base:
                params = base['parameters']
                report_lines.append(f"Размерность: {params.get('n_dim', 'N/A')}")
                report_lines.append(f"Размер роя: {params.get('swarm_size', 'N/A')}")
                report_lines.append(f"Коэффициент инерции (w): {params.get('w', 'N/A')}")
                report_lines.append(f"Когнитивный коэффициент (c1): {params.get('c1', 'N/A')}")
                report_lines.append(f"Социальный коэффициент (c2): {params.get('c2', 'N/A')}")
            report_lines.append(f"Лучшее значение f(x): {base.get('best_value', 'N/A'):.6f}")
            report_lines.append(f"Число итераций: {base.get('iterations', 'N/A')}")
            report_lines.append(f"Сошелся: {'Да' if base.get('converged', False) else 'Нет'}")
            report_lines.append(f"Время выполнения: {base.get('total_time', 'N/A'):.2f} сек")
        report_lines.append("")

    # Добавляем результаты экспериментов с параметрами
    param_experiments = ['swarm_size', 'w', 'c1', 'c2', 'v_max', 'dimensions', 'c1_c2_balance']

    for param in param_experiments:
        if param in results:
            report_lines.append(f"2. ЭКСПЕРИМЕНТ: Влияние параметра '{param}'")
            report_lines.append("-" * 40)

            param_results = results[param]
            if isinstance(param_results, dict):
                for param_value, res in param_results.items():
                    if isinstance(res, dict) and 'best_value' in res:
                        report_lines.append(f"  {param} = {param_value}:")
                        report_lines.append(f"    Лучшее f(x) = {res.get('best_value', 'N/A'):.6f}")
                        report_lines.append(f"    Итераций = {res.get('iterations', 'N/A')}")
                        report_lines.append(f"    Успешность = {res.get('success_rate', 0) * 100:.1f}%")
                        report_lines.append("")
            report_lines.append("")

    # Вывод рекомендаций
    report_lines.append("3. РЕКОМЕНДАЦИИ ПО ВЫБОРУ ПАРАМЕТРОВ PSO")
    report_lines.append("-" * 40)
    report_lines.append("1. Размер роя: 20-50 частиц (больше для сложных задач)")
    report_lines.append("2. Коэффициент инерции w: 0.7-0.9 (уменьшать со временем)")
    report_lines.append("3. Коэффициенты c1, c2: 1.5-2.0 (баланс личного/коллективного опыта)")
    report_lines.append("4. Максимальная скорость: 10-20% от диапазона поиска")
    report_lines.append("5. Для функции Растригина важны большие значения c1, c2")
    report_lines.append("   для преодоления локальных минимумов")
    report_lines.append("")

    report_lines.append("4. ВЫВОДЫ")
    report_lines.append("-" * 40)
    report_lines.append("1. PSO эффективен для многомодальных функций типа Растригина")
    report_lines.append("2. Баланс между exploration (исследование) и exploitation (использование)")
    report_lines.append("   критически важен для успеха алгоритма")
    report_lines.append("3. Разнообразие роя уменьшается со временем - необходимы механизмы")
    report_lines.append("   его поддержания для предотвращения преждевременной сходимости")
    report_lines.append("4. PSO показывает хорошую масштабируемость до 10-20 измерений")
    report_lines.append("")

    report_lines.append("=" * 70)
    report_lines.append("КОНЕЦ ОТЧЕТА")
    report_lines.append("=" * 70)

    # Сохраняем отчет в файл
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"\n✅ Отчет сохранен в файл: {filename}")

        # Также выводим краткую версию в консоль
        print("\n📋 КРАТКИЙ ОТЧЕТ:")
        print("-" * 40)
        for line in report_lines[:30]:  # Первые 30 строк
            print(line)

        return filename
    except Exception as e:
        print(f"\n❌ Ошибка при сохранении отчета: {e}")
        return None


# ==================== ГЛАВНЫЕ ФУНКЦИИ ====================

def run_complete_pso_experiment():
    """Запуск полного набора экспериментов с PSO."""
    print("🚀 ЗАПУСК ЛАБОРАТОРНОЙ РАБОТЫ ПО АЛГОРИТМУ PSO")
    print("=" * 70)

    all_results = {}

    # 1. Визуализация функции Растригина
    print("\n📊 1. Визуализация функции Растригина...")
    fig = visualize_rastrigin_2d()

    # 2. Базовый запуск PSO
    print("\n🐝 2. Базовый запуск алгоритма оптимизации роем частиц...")
    base_result = pso_algorithm(n_dim=2, **PSO_BASE_PARAMS)
    print(f"   Лучшее f(x) = {base_result['best_value']:.6f}")
    print(f"   Итераций = {base_result['iterations']}")
    print(f"   Сошелся = {base_result['converged']}")

    all_results['base_run'] = base_result

    # 3. Визуализация траекторий частиц (только для 2D)
    if base_result['parameters']['n_dim'] == 2:
        print("\n📈 3. Визуализация траекторий частиц...")
        try:
            trajectory_fig = plot_particle_trajectories(base_result)
        except Exception as e:
            print(f"   ⚠️  Ошибка при визуализации траекторий: {e}")

    # 4. Эксперимент с размером роя
    print("\n📈 4. Эксперимент: Влияние размера роя...")
    swarm_results, fig_swarm = experiment_swarm_size(n_dim=2, n_runs=3)
    all_results['swarm_size'] = swarm_results

    # 5. Эксперимент с коэффициентом инерции
    print("\n🌀 5. Эксперимент: Влияние коэффициента инерции...")
    inertia_results, fig_inertia = experiment_inertia_weight(n_dim=2, n_runs=3)
    all_results['w'] = inertia_results

    # 6. Эксперимент с когнитивным коэффициентом
    print("\n🧠 6. Эксперимент: Влияние когнитивного коэффициента...")
    cognitive_results, fig_cognitive = experiment_cognitive_coefficient(n_dim=2, n_runs=3)
    all_results['c1'] = cognitive_results

    # 7. Эксперимент с социальным коэффициента
    print("\n👥 7. Эксперимент: Влияние социального коэффициента...")
    social_results, fig_social = experiment_social_coefficient(n_dim=2, n_runs=3)
    all_results['c2'] = social_results

    # 8. Эксперимент с максимальной скоростью
    print("\n⚡ 8. Эксперимент: Влияние максимальной скорости...")
    vmax_results, fig_vmax = experiment_vmax(n_dim=2, n_runs=3)
    all_results['v_max'] = vmax_results

    # 9. Эксперимент с балансом c1 и c2
    print("\n⚖️  9. Эксперимент: Сравнение баланса c1 и c2...")
    balance_results, fig_balance = experiment_compare_c1_c2(n_dim=2, n_runs=3)
    all_results['c1_c2_balance'] = balance_results

    # 10. Эксперимент с размерностью
    print("\n📏 10. Эксперимент: Влияние размерности задачи...")
    dim_results, fig_dim = experiment_dimensions_pso(n_runs=3)
    all_results['dimensions'] = dim_results

    # 11. Создаем график сходимости для базового запуска
    print("\n📈 11. Построение графиков сходимости...")
    try:
        convergence_fig = plot_convergence_curves_pso({'Базовый PSO': base_result})
    except Exception as e:
        print(f"   ⚠️  Ошибка при построении графиков сходимости: {e}")

    # 12. Сохраняем результаты и генерируем отчет
    print("\n💾 12. Сохранение результатов и генерация отчета...")
    save_results_to_file(all_results)
    generate_report(all_results)

    print(f"\n{'=' * 70}")
    print("✅ ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА")
    print("=" * 70)
    print("\n📁 Созданные файлы:")
    print("  • rastrigin_function_pso.png - визуализация функции")
    print("  • pso_convergence_curves.png - графики сходимости")
    print("  • pso_particle_trajectories.png - траектории частиц")
    print("  • pso_swarm_size_experiment.png - влияние размера роя")
    print("  • pso_inertia_weight_experiment.png - влияние коэффициента инерции")
    print("  • pso_cognitive_coefficient_experiment.png - влияние c1")
    print("  • pso_social_coefficient_experiment.png - влияние c2")
    print("  • pso_v_max_experiment.png - влияние максимальной скорости")
    print("  • pso_c1_c2_comparison.png - сравнение баланса c1 и c2")
    print("  • pso_dimension_experiment.png - влияние размерности")
    print("  • pso_experiment_results.json - все результаты")
    print("  • pso_lab_report.txt - текстовый отчет")

    return all_results


def run_quick_pso_experiment():
    """Упрощенный запуск для быстрого тестирования PSO."""
    print("🚀 БЫСТРЫЙ ЗАПУСК АЛГОРИТМА PSO")
    print("=" * 50)

    # 1. Визуализация функции
    visualize_rastrigin_2d()

    # 2. Несколько запусков с разными параметрами
    test_configs = {
        "Базовый PSO": PSO_BASE_PARAMS,
        "Большой рой": {**PSO_BASE_PARAMS, "swarm_size": 50},
        "Высокая инерция": {**PSO_BASE_PARAMS, "w": 0.9},
        "Низкая скорость": {**PSO_BASE_PARAMS, "v_max": 0.5},
        "Личный опыт (высокий c1)": {**PSO_BASE_PARAMS, "c1": 2.5, "c2": 0.5},
        "Коллективный опыт (высокий c2)": {**PSO_BASE_PARAMS, "c1": 0.5, "c2": 2.5},
    }

    results = {}
    for name, params in test_configs.items():
        print(f"\n🐝 {name}...")
        result = pso_algorithm(n_dim=2, **params)
        results[name] = result
        print(f"   f(x) = {result['best_value']:.6f}, итераций = {result['iterations']}")

    # 3. График сходимости
    fig = plot_convergence_curves_pso(results, "Сравнение разных конфигураций PSO")

    # 4. Визуализация траекторий для базового случая
    if "Базовый PSO" in results:
        plot_particle_trajectories(results["Базовый PSO"])

    return results


def interactive_pso_experiment():
    """Интерактивный запуск PSO с пользовательскими параметрами."""
    print("\n🔧 ИНТЕРАКТИВНЫЙ ЭКСПЕРИМЕНТ С PSO")
    print("=" * 50)

    # Параметры от пользователя
    n_dim = int(input("Размерность задачи (2-20, рекомендовано 2): ") or "2")
    swarm_size = int(input(f"Размер роя (рекомендовано 30): ") or "30")
    iterations = int(input(f"Макс. число итераций (рекомендовано 100): ") or "100")
    w = float(input(f"Коэффициент инерции w (0-2, рекомендовано 0.7): ") or "0.7")
    c1 = float(input(f"Когнитивный коэффициент c1 (0-3, рекомендовано 1.5): ") or "1.5")
    c2 = float(input(f"Социальный коэффициент c2 (0-3, рекомендовано 1.5): ") or "1.5")
    v_max = float(input(f"Максимальная скорость (рекомендовано 1.0): ") or "1.0")

    # Выбор стратегии границ
    print("\nСтратегии обработки границ:")
    print("1. reflect - отражение от границ (по умолчанию)")
    print("2. absorb - поглощение (остановка на границе)")
    print("3. cyclic - циклическое возвращение")
    boundary_choice = input("Выберите стратегию (1-3): ").strip()

    boundary_strategies = {"1": "reflect", "2": "absorb", "3": "cyclic"}
    boundary_strategy = boundary_strategies.get(boundary_choice, "reflect")

    # Сбор параметров
    pso_params = {
        "swarm_size": swarm_size,
        "iterations": iterations,
        "w": w,
        "c1": c1,
        "c2": c2,
        "v_max": v_max,
        "boundary_strategy": boundary_strategy,
    }

    # Запуск
    print(f"\n🐝 Запуск PSO с параметрами:")
    print(f"   Размерность: {n_dim}")
    print(f"   Размер роя: {swarm_size}")
    print(f"   Итераций: {iterations}")
    print(f"   w: {w}")
    print(f"   c1: {c1}")
    print(f"   c2: {c2}")
    print(f"   v_max: {v_max}")
    print(f"   Стратегия границ: {boundary_strategy}")

    result = pso_algorithm(n_dim=n_dim, **pso_params)

    # Вывод результатов
    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    print(f"   Лучшая позиция: {result['best_position']}")
    print(f"   Лучшее f(x) = {result['best_value']:.6f}")
    print(f"   Итераций = {result['iterations']}")
    print(f"   Сошелся = {result['converged']}")
    print(f"   Время = {result['total_time']:.2f} сек")

    # График сходимости
    fig = plot_convergence_curves_pso({"Интерактивный запуск": result})

    # Визуализация траекторий для 2D
    if n_dim == 2:
        plot_particle_trajectories(result)

    return result


def test_specific_pso_parameter():
    """Тестирование конкретного параметра PSO."""
    print("\n🔍 ТЕСТИРОВАНИЕ КОНКРЕТНОГО ПАРАМЕТРА PSO")
    print("=" * 50)
    print("Выберите параметр для тестирования:")
    print("1. Размер роя")
    print("2. Коэффициент инерции (w)")
    print("3. Когнитивный коэффициент (c1)")
    print("4. Социальный коэффициент (c2)")
    print("5. Максимальная скорость (v_max)")
    print("6. Размерность задачи")
    print("7. Сравнение баланса c1 и c2")

    param_choice = input("Введите номер (1-7): ").strip()

    n_dim = 2
    n_runs = 3

    if param_choice == "1":
        print("\n📊 Тестирование: Размер роя")
        n_dim = int(input("Размерность задачи (по умолчанию 2): ") or "2")
        n_runs = int(input("Количество запусков (по умолчанию 3): ") or "3")
        results, fig = experiment_swarm_size(n_dim=n_dim, n_runs=n_runs)

    elif param_choice == "2":
        print("\n📊 Тестирование: Коэффициент инерции")
        n_dim = int(input("Размерность задачи (по умолчанию 2): ") or "2")
        n_runs = int(input("Количество запусков (по умолчанию 3): ") or "3")
        results, fig = experiment_inertia_weight(n_dim=n_dim, n_runs=n_runs)

    elif param_choice == "3":
        print("\n📊 Тестирование: Когнитивный коэффициент")
        n_dim = int(input("Размерность задачи (по умолчанию 2): ") or "2")
        n_runs = int(input("Количество запусков (по умолчанию 3): ") or "3")
        results, fig = experiment_cognitive_coefficient(n_dim=n_dim, n_runs=n_runs)

    elif param_choice == "4":
        print("\n📊 Тестирование: Социальный коэффициент")
        n_dim = int(input("Размерность задачи (по умолчанию 2): ") or "2")
        n_runs = int(input("Количество запусков (по умолчанию 3): ") or "3")
        results, fig = experiment_social_coefficient(n_dim=n_dim, n_runs=n_runs)

    elif param_choice == "5":
        print("\n📊 Тестирование: Максимальная скорость")
        n_dim = int(input("Размерность задачи (по умолчанию 2): ") or "2")
        n_runs = int(input("Количество запусков (по умолчанию 3): ") or "3")
        results, fig = experiment_vmax(n_dim=n_dim, n_runs=n_runs)

    elif param_choice == "6":
        print("\n📊 Тестирование: Размерность задачи")
        n_runs = int(input("Количество запусков (по умолчанию 3): ") or "3")
        results, fig = experiment_dimensions_pso(n_runs=n_runs)

    elif param_choice == "7":
        print("\n📊 Тестирование: Сравнение баланса c1 и c2")
        n_dim = int(input("Размерность задачи (по умолчанию 2): ") or "2")
        n_runs = int(input("Количество запусков (по умолчанию 3): ") or "3")
        results, fig = experiment_compare_c1_c2(n_dim=n_dim, n_runs=n_runs)

    else:
        print("❌ Неверный выбор!")
        return None

    print(f"\n✅ Тестирование завершено!")
    print(f"📁 График сохранен в файл")
    return results


# ==================== ГЛАВНОЕ МЕНЮ ====================

if __name__ == "__main__":
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА: АЛГОРИТМ ОПТИМИЗАЦИИ РОЕМ ЧАСТИЦ (PSO)")
    print("=" * 60)
    print("Выберите режим работы:")
    print("1. Полный набор экспериментов (все графики + анализ)")
    print("2. Быстрый тест (несколько конфигураций)")
    print("3. Интерактивный эксперимент")
    print("4. Только визуализация функции")
    print("5. Тест конкретного параметра")
    print("6. Загрузить результаты из файла и сгенерировать отчет")

    choice = input("Введите номер (1-6): ").strip()

    if choice == "1":
        # Полный эксперимент
        all_results = run_complete_pso_experiment()

    elif choice == "2":
        # Быстрый тест
        results = run_quick_pso_experiment()

    elif choice == "3":
        # Интерактивный
        result = interactive_pso_experiment()

    elif choice == "4":
        # Только визуализация
        print("\n📊 ВИЗУАЛИЗАЦИЯ ФУНКЦИИ РАСТРИГИНА")
        print("=" * 40)
        fig = visualize_rastrigin_2d()

    elif choice == "5":
        # Тест конкретного параметра
        results = test_specific_pso_parameter()

    elif choice == "6":
        # Загрузка из файла
        filename = input(
            "Имя файла с результатами (по умолчанию pso_experiment_results.json): ") or "pso_experiment_results.json"
        results = load_results_from_file(filename)
        if results:
            generate_report(results)

    else:
        print("❌ Неверный выбор. Запускаю быстрый тест...")
        results = run_quick_pso_experiment()

    print("\n" + "=" * 60)
    print("РАБОТА ЗАВЕРШЕНА")
    print("=" * 60)