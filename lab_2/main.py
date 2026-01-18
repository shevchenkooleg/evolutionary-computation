import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import time
from tqdm import tqdm
import re
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# ==================== КОНСТАНТЫ И ПАРАМЕТРЫ ДЛЯ ЭКСПЕРИМЕНТОВ ====================

# 1. ПАРАМЕТРЫ ФУНКЦИИ РАСТРИГИНА
RASTRIGIN_PARAMS = {
    "A": 10,  # Параметр функции
    "dimensions": [2, 5, 10, 20],  # Размерности для исследования
    "bounds": (-5.12, 5.12),  # Границы поиска
}

# 2. БАЗОВЫЕ ПАРАМЕТРЫ ГЕНЕТИЧЕСКОГО АЛГОРИТМА
GA_BASE_PARAMS = {
    "population_size": 50,  # N
    "generations": 100,  # G_max
    "crossover_prob": 0.8,  # p_c
    "mutation_prob": 0.1,  # p_m
    "mutation_strength": 0.5,  # σ
    "tournament_size": 3,  # k_tour
    "elite_count": 2,  # Количество элитных особей
    "alpha": 0.5,  # Параметр для арифметического кроссовера
}

# 3. ЗНАЧЕНИЯ ПАРАМЕТРОВ ДЛЯ ИССЛЕДОВАНИЯ
PARAM_SWEEPS = {
    "population_size": [10, 25, 50, 100],  # N
    "mutation_prob": [0.01, 0.05, 0.1, 0.2, 0.4],  # p_m
    "mutation_strength": [0.05, 0.1, 0.2, 0.5, 1.0],  # σ
    "crossover_prob": [0.5, 0.7, 0.9, 1.0],  # p_c
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


def initialize_population(pop_size, n, low=-5.12, high=5.12):
    """
    Инициализация случайной популяции.

    Parameters:
    -----------
    pop_size : int
        Размер популяции
    n : int
        Размерность задачи
    low, high : float
        Границы поиска

    Returns:
    --------
    numpy.ndarray : Популяция размера (pop_size, n)
    """
    return np.random.uniform(low, high, size=(pop_size, n))


def fitness_function(population, A=10):
    """
    Вычисление приспособленности особей.
    Для минимизации: чем меньше f(x), тем выше приспособленность.

    Parameters:
    -----------
    population : numpy.ndarray
        Популяция особей
    A : float
        Параметр функции Растригина

    Returns:
    --------
    tuple : (fitness, f_values)
    """
    # Вычисляем значения функции для всех особей
    f_values = np.array([rastrigin_function(ind, A) for ind in population])

    # Преобразуем в приспособленность (больше = лучше)
    # Используем: fitness = 1 / (1 + f(x))
    fitness = 1.0 / (1.0 + np.abs(f_values))

    return fitness, f_values


def tournament_selection(population, fitness, tournament_size=3):
    """
    Турнирный отбор родителей.

    Parameters:
    -----------
    population : numpy.ndarray
        Популяция
    fitness : numpy.ndarray
        Значения приспособленности
    tournament_size : int
        Размер турнира

    Returns:
    --------
    numpy.ndarray : Выбранный родитель
    """
    # Выбираем случайных участников турнира
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)

    # Выбираем лучшего (с максимальной приспособленностью)
    best_idx = selected_indices[np.argmax(fitness[selected_indices])]

    return population[best_idx].copy()


def arithmetic_crossover(parent1, parent2, alpha=0.5):
    """
    Арифметический кроссовер (взвешенное среднее).

    Parameters:
    -----------
    parent1, parent2 : numpy.ndarray
        Родительские особи
    alpha : float
        Параметр кроссовера (0 < alpha < 1)

    Returns:
    --------
    tuple : Два потомка
    """
    # Создаем потомков как линейные комбинации родителей
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2

    return child1, child2


def gaussian_mutation(individual, mutation_prob, mutation_strength, bounds=(-5.12, 5.12)):
    """
    Гауссова мутация.

    Parameters:
    -----------
    individual : numpy.ndarray
        Особь для мутации
    mutation_prob : float
        Вероятность мутации каждого гена
    mutation_strength : float
        Сила мутации (σ в нормальном распределении)
    bounds : tuple
        Границы поиска

    Returns:
    --------
    numpy.ndarray : Мутированная особь
    """
    mutated = individual.copy()

    # Маска для генов, которые будут мутировать
    mask = np.random.rand(len(individual)) < mutation_prob

    if np.any(mask):
        # Добавляем гауссов шум
        noise = np.random.normal(0, mutation_strength, len(individual))
        mutated[mask] += noise[mask]

        # Ограничиваем значения границами
        mutated = np.clip(mutated, bounds[0], bounds[1])

    return mutated


def genetic_algorithm(n_dim=2, **ga_params):
    """
    Основная функция генетического алгоритма.

    Parameters:
    -----------
    n_dim : int
        Размерность задачи
    ga_params : dict
        Параметры ГА

    Returns:
    --------
    dict : Результаты оптимизации
    """
    # Извлечение параметров
    pop_size = ga_params.get("population_size", 50)
    generations = ga_params.get("generations", 100)
    p_crossover = ga_params.get("crossover_prob", 0.8)
    p_mutation = ga_params.get("mutation_prob", 0.1)
    mutation_strength = ga_params.get("mutation_strength", 0.5)
    tournament_size = ga_params.get("tournament_size", 3)
    elite_count = ga_params.get("elite_count", 2)
    crossover_alpha = ga_params.get("alpha", 0.5)
    A = ga_params.get("A", 10)
    bounds = ga_params.get("bounds", (-5.12, 5.12))

    # Параметры остановки
    max_generations = ga_params.get("max_generations", 100)
    target_fitness = ga_params.get("target_fitness", 1e-4)
    stagnation_gen = ga_params.get("stagnation_generations", 20)

    # Инициализация
    start_time = time.time()
    population = initialize_population(pop_size, n_dim, bounds[0], bounds[1])

    # История
    history = {
        'best_fitness': [],  # Лучшая приспособленность
        'avg_fitness': [],  # Средняя приспособленность
        'best_f_value': [],  # Лучшее значение функции
        'avg_f_value': [],  # Среднее значение функции
        'best_individual': [],  # Лучшая особь
        'population_diversity': [],  # Разнообразие популяции
        'generation_time': [],  # Время на поколение
    }

    # Переменные для отслеживания стагнации
    best_f_value_global = float('inf')
    stagnation_counter = 0
    converged = False

    # Основной цикл эволюции
    for gen in range(max_generations):
        gen_start_time = time.time()

        # 1. Оценка приспособленности
        fitness, f_values = fitness_function(population, A)

        # 2. Сохранение статистики
        best_idx = np.argmax(fitness)
        best_f = fitness[best_idx]
        best_f_val = f_values[best_idx]
        avg_f = np.mean(fitness)
        avg_f_val = np.mean(f_values)

        # Вычисление разнообразия (среднее расстояние между особями)
        if pop_size > 1:
            # Используем среднее попарное евклидово расстояние
            diversity = 0
            count = 0
            for i in range(pop_size):
                for j in range(i + 1, pop_size):
                    diversity += np.linalg.norm(population[i] - population[j])
                    count += 1
            diversity = diversity / count if count > 0 else 0
        else:
            diversity = 0

        # Сохранение в историю
        history['best_fitness'].append(best_f)
        history['avg_fitness'].append(avg_f)
        history['best_f_value'].append(best_f_val)
        history['avg_f_value'].append(avg_f_val)
        history['best_individual'].append(population[best_idx].copy())
        history['population_diversity'].append(diversity)

        # 3. Проверка критериев остановки
        # а) Достигнута целевая точность
        if best_f_val < target_fitness:
            converged = True
            break

        # б) Проверка стагнации
        if best_f_val < best_f_value_global - 1e-6:  # Небольшое улучшение
            best_f_value_global = best_f_val
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter >= stagnation_gen:
            converged = False  # Сошлось к субоптимуму
            break

        # 4. Формирование новой популяции
        new_population = []

        # а) Элитизм: сохраняем лучших особей
        elite_indices = np.argsort(fitness)[-elite_count:][::-1]
        for idx in elite_indices:
            new_population.append(population[idx].copy())

        # б) Создание потомков
        while len(new_population) < pop_size:
            # Селекция родителей
            parent1 = tournament_selection(population, fitness, tournament_size)
            parent2 = tournament_selection(population, fitness, tournament_size)

            # Кроссовер
            if np.random.rand() < p_crossover:
                child1, child2 = arithmetic_crossover(parent1, parent2, crossover_alpha)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Мутация
            child1 = gaussian_mutation(child1, p_mutation, mutation_strength, bounds)
            child2 = gaussian_mutation(child2, p_mutation, mutation_strength, bounds)

            # Добавляем потомков в новую популяцию
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        # Обрезаем, если добавили слишком много
        new_population = np.array(new_population[:pop_size])
        population = new_population

        # Время поколения
        history['generation_time'].append(time.time() - gen_start_time)

    # Формирование результатов
    total_time = time.time() - start_time

    # Находим лучший результат за всю историю
    if len(history['best_f_value']) > 0:
        best_gen_idx = np.argmin(history['best_f_value'])
        best_individual = history['best_individual'][best_gen_idx]
        best_f_value = history['best_f_value'][best_gen_idx]
    else:
        best_individual = population[0]
        best_f_value = rastrigin_function(best_individual, A)

    result = {
        'best_individual': best_individual,
        'best_f_value': best_f_value,
        'converged': converged,
        'generations': gen + 1,
        'total_time': total_time,
        'history': history,
        'parameters': {
            'n_dim': n_dim,
            'pop_size': pop_size,
            'p_crossover': p_crossover,
            'p_mutation': p_mutation,
            'mutation_strength': mutation_strength,
            'elite_count': elite_count,
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
    plt.savefig('rastrigin_function.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def plot_convergence_curves(results_dict, title="Сравнение сходимости ГА"):
    """Построение графиков сходимости для разных конфигураций."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    colors = plt.cm.rainbow(np.linspace(0, 1, len(results_dict)))

    for (label, result), color in zip(results_dict.items(), colors):
        history = result['history']
        if len(history['best_f_value']) == 0:
            continue

        generations = list(range(1, len(history['best_f_value']) + 1))

        # График 1: Лучшее значение функции
        axes[0, 0].semilogy(generations, history['best_f_value'],
                            label=label, color=color, linewidth=2)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Поколение', fontsize=12)
        axes[0, 0].set_ylabel('Лучшее f(x) (log scale)', fontsize=12)
        axes[0, 0].set_title('Сходимость лучшего значения', fontsize=13)
        axes[0, 0].grid(True, alpha=0.3)

        # График 2: Среднее значение функции
        axes[0, 1].plot(generations, history['avg_f_value'],
                        label=label, color=color, linewidth=2)
        axes[0, 1].set_xlabel('Поколение', fontsize=12)
        axes[0, 1].set_ylabel('Среднее f(x)', fontsize=12)
        axes[0, 1].set_title('Эволюция среднего значения', fontsize=13)
        axes[0, 1].grid(True, alpha=0.3)

        # График 3: Разнообразие популяции
        axes[1, 0].plot(generations, history['population_diversity'],
                        label=label, color=color, linewidth=2)
        axes[1, 0].set_xlabel('Поколение', fontsize=12)
        axes[1, 0].set_ylabel('Разнообразие популяции', fontsize=12)
        axes[1, 0].set_title('Динамика разнообразия', fontsize=13)
        axes[1, 0].grid(True, alpha=0.3)

        # График 4: Время выполнения
        if ('generation_time' in history and
                len(history['generation_time']) > 0 and
                len(history['generation_time']) == len(history['best_f_value'])):

            cumulative_time = np.cumsum(history['generation_time'])
            axes[1, 1].plot(cumulative_time, history['best_f_value'],
                            label=label, color=color, linewidth=2)
            axes[1, 1].set_xlabel('Время (сек)', fontsize=12)
            axes[1, 1].set_ylabel('Лучшее f(x)', fontsize=12)
            axes[1, 1].set_title('Сходимость по времени', fontsize=13)
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Если времена нет или размеры не совпадают, скрываем этот график
            axes[1, 1].text(0.5, 0.5, 'Данные о времени недоступны',
                            ha='center', va='center', fontsize=12,
                            transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Сходимость по времени', fontsize=13)

    # Добавляем легенду только один раз
    axes[0, 0].legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    plt.savefig('ga_convergence_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def plot_parameter_sweep_results(sweep_results, param_name, title=None):
    """Визуализация результатов сканирования параметра."""
    if title is None:
        title = f'Влияние параметра {param_name} на эффективность ГА'

    param_values = list(sweep_results.keys())
    best_values = [r['best_f_value'] for r in sweep_results.values()]

    # Для вычисления средних значений
    avg_values = []
    for val in param_values:
        if 'avg_f_value' in sweep_results[val]:
            avg_values.append(sweep_results[val]['avg_f_value'])
        else:
            # Вычисляем среднее из истории
            if 'all_runs' in sweep_results[val] and len(sweep_results[val]['all_runs']) > 0:
                history_avg = np.mean(sweep_results[val]['all_runs'][0]['history']['avg_f_value'])
            else:
                history_avg = sweep_results[val]['best_f_value']
            avg_values.append(history_avg)

    converged = [r.get('converged', True) for r in sweep_results.values()]
    iterations = [r.get('generations', 100) for r in sweep_results.values()]

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

    # График 2: Число поколений vs параметр
    ax2 = axes[0, 1]
    ax2.plot(param_values, iterations, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel(param_name, fontsize=12)
    ax2.set_ylabel('Число поколений', fontsize=12)
    ax2.set_title('Скорость сходимости', fontsize=13)
    ax2.grid(True, alpha=0.3)

    # График 3: Среднее значение vs параметр
    ax3 = axes[1, 0]
    ax3.plot(param_values, avg_values, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel(param_name, fontsize=12)
    ax3.set_ylabel('Среднее f(x)', fontsize=12)
    ax3.set_title('Среднее качество популяции', fontsize=13)
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
    filename = f'{safe_param_name}_experiment.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    return fig


# ==================== ЭКСПЕРИМЕНТЫ ====================

def experiment_population_size(n_dim=2, n_runs=3):
    """Эксперимент: влияние размера популяции."""
    print(f"\n{'=' * 60}")
    print("ЭКСПЕРИМЕНТ: Влияние размера популяции N")
    print(f"{'=' * 60}")

    results = {}
    param_values = PARAM_SWEEPS["population_size"]

    for pop_size in tqdm(param_values, desc="Размер популяции"):
        run_results = []

        for run in range(n_runs):
            ga_params = GA_BASE_PARAMS.copy()
            ga_params["population_size"] = pop_size

            result = genetic_algorithm(n_dim=n_dim, **ga_params)
            run_results.append(result)

        # Агрегируем результаты по запускам
        avg_best = np.mean([r['best_f_value'] for r in run_results])
        avg_generations = np.mean([r['generations'] for r in run_results])
        success_rate = np.mean([r['converged'] for r in run_results])

        results[pop_size] = {
            'best_f_value': avg_best,
            'avg_f_value': np.mean([np.mean(r['history']['avg_f_value']) for r in run_results]),
            'generations': avg_generations,
            'converged': success_rate > 0.5,
            'success_rate': success_rate,
            'all_runs': run_results
        }

        print(f"N={pop_size:3d}: f={avg_best:.6f}, gen={avg_generations:.1f}, "
              f"success={success_rate * 100:.0f}%")

    # Визуализация
    fig = plot_parameter_sweep_results(results, "population_size",
                                       "Влияние размера популяции на эффективность ГА")
    return results, fig


def experiment_mutation_prob(n_dim=2, n_runs=3):
    """Эксперимент: влияние вероятности мутации."""
    print(f"\n{'=' * 60}")
    print("ЭКСПЕРИМЕНТ: Влияние вероятности мутации p_m")
    print(f"{'=' * 60}")

    results = {}
    param_values = PARAM_SWEEPS["mutation_prob"]

    for p_m in tqdm(param_values, desc="Вероятность мутации"):
        run_results = []

        for run in range(n_runs):
            ga_params = GA_BASE_PARAMS.copy()
            ga_params["mutation_prob"] = p_m
            ga_params["mutation_strength"] = 0.2  # Фиксируем для чистоты эксперимента

            result = genetic_algorithm(n_dim=n_dim, **ga_params)
            run_results.append(result)

        avg_best = np.mean([r['best_f_value'] for r in run_results])
        avg_generations = np.mean([r['generations'] for r in run_results])
        success_rate = np.mean([r['converged'] for r in run_results])

        results[p_m] = {
            'best_f_value': avg_best,
            'avg_f_value': np.mean([np.mean(r['history']['avg_f_value']) for r in run_results]),
            'generations': avg_generations,
            'converged': success_rate > 0.5,
            'success_rate': success_rate,
            'all_runs': run_results
        }

        print(f"p_m={p_m:.2f}: f={avg_best:.6f}, gen={avg_generations:.1f}, "
              f"success={success_rate * 100:.0f}%")

    # Визуализация
    fig = plot_parameter_sweep_results(results, "mutation_prob",
                                       "Влияние вероятности мутации на эффективность ГА")
    return results, fig


def experiment_mutation_strength(n_dim=2, n_runs=3):
    """Эксперимент: влияние силы мутации."""
    print(f"\n{'=' * 60}")
    print("ЭКСПЕРИМЕНТ: Влияние силы мутации σ")
    print(f"{'=' * 60}")

    results = {}
    param_values = PARAM_SWEEPS["mutation_strength"]

    for sigma in tqdm(param_values, desc="Сила мутации"):
        run_results = []

        for run in range(n_runs):
            ga_params = GA_BASE_PARAMS.copy()
            ga_params["mutation_strength"] = sigma
            ga_params["mutation_prob"] = 0.1  # Фиксируем для чистоты эксперимента

            result = genetic_algorithm(n_dim=n_dim, **ga_params)
            run_results.append(result)

        avg_best = np.mean([r['best_f_value'] for r in run_results])
        avg_generations = np.mean([r['generations'] for r in run_results])
        success_rate = np.mean([r['converged'] for r in run_results])

        results[sigma] = {
            'best_f_value': avg_best,
            'avg_f_value': np.mean([np.mean(r['history']['avg_f_value']) for r in run_results]),
            'generations': avg_generations,
            'converged': success_rate > 0.5,
            'success_rate': success_rate,
            'all_runs': run_results
        }

        print(f"σ={sigma:.2f}: f={avg_best:.6f}, gen={avg_generations:.1f}, "
              f"success={success_rate * 100:.0f}%")

    # Визуализация
    fig = plot_parameter_sweep_results(results, "mutation_strength",
                                       "Влияние силы мутации на эффективность ГА")
    return results, fig


def experiment_crossover_prob(n_dim=2, n_runs=3):
    """Эксперимент: влияние вероятности кроссовера."""
    print(f"\n{'=' * 60}")
    print("ЭКСПЕРИМЕНТ: Влияние вероятности кроссовера p_c")
    print(f"{'=' * 60}")

    results = {}
    param_values = PARAM_SWEEPS["crossover_prob"]

    for p_c in tqdm(param_values, desc="Вероятность кроссовера"):
        run_results = []

        for run in range(n_runs):
            ga_params = GA_BASE_PARAMS.copy()
            ga_params["crossover_prob"] = p_c

            result = genetic_algorithm(n_dim=n_dim, **ga_params)
            run_results.append(result)

        avg_best = np.mean([r['best_f_value'] for r in run_results])
        avg_generations = np.mean([r['generations'] for r in run_results])
        success_rate = np.mean([r['converged'] for r in run_results])

        results[p_c] = {
            'best_f_value': avg_best,
            'avg_f_value': np.mean([np.mean(r['history']['avg_f_value']) for r in run_results]),
            'generations': avg_generations,
            'converged': success_rate > 0.5,
            'success_rate': success_rate,
            'all_runs': run_results
        }

        print(f"p_c={p_c:.2f}: f={avg_best:.6f}, gen={avg_generations:.1f}, "
              f"success={success_rate * 100:.0f}%")

    # Визуализация
    fig = plot_parameter_sweep_results(results, "crossover_prob",
                                       "Влияние вероятности кроссовера на эффективность ГА")
    return results, fig


def experiment_dimensions(n_runs=3):
    """Эксперимент: влияние размерности задачи."""
    print(f"\n{'=' * 60}")
    print("ЭКСПЕРИМЕНТ: Влияние размерности задачи")
    print(f"{'=' * 60}")

    results = {}
    dimensions = RASTRIGIN_PARAMS["dimensions"]

    for n_dim in tqdm(dimensions, desc="Размерность"):
        run_results = []

        for run in range(n_runs):
            # Корректируем параметры для большей размерности
            ga_params = GA_BASE_PARAMS.copy()
            if n_dim > 5:
                # Увеличиваем популяцию для больших размерностей
                ga_params["population_size"] = 100
                ga_params["generations"] = 200

            result = genetic_algorithm(n_dim=n_dim, **ga_params)
            run_results.append(result)

        avg_best = np.mean([r['best_f_value'] for r in run_results])
        avg_generations = np.mean([r['generations'] for r in run_results])
        success_rate = np.mean([r['converged'] for r in run_results])

        results[n_dim] = {
            'best_f_value': avg_best,
            'avg_f_value': np.mean([np.mean(r['history']['avg_f_value']) for r in run_results]),
            'generations': avg_generations,
            'converged': success_rate > 0.5,
            'success_rate': success_rate,
            'all_runs': run_results
        }

        print(f"n={n_dim:2d}: f={avg_best:.6f}, gen={avg_generations:.1f}, "
              f"success={success_rate * 100:.0f}%")

    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Влияние размерности задачи на эффективность ГА',
                 fontsize=16, fontweight='bold')

    dimensions_list = list(results.keys())
    best_values = [results[d]['best_f_value'] for d in dimensions_list]
    avg_values = [results[d]['avg_f_value'] for d in dimensions_list]
    generations = [results[d]['generations'] for d in dimensions_list]
    success_rates = [results[d]['success_rate'] for d in dimensions_list]

    # График 1: Качество решения
    axes[0, 0].plot(dimensions_list, best_values, 'ro-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Размерность (n)', fontsize=12)
    axes[0, 0].set_ylabel('Лучшее f(x)', fontsize=12)
    axes[0, 0].set_title('Качество решения', fontsize=13)
    axes[0, 0].grid(True, alpha=0.3)
    if max(best_values) > 0:
        axes[0, 0].set_yscale('log')

    # График 2: Требуемое число поколений
    axes[0, 1].plot(dimensions_list, generations, 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Размерность (n)', fontsize=12)
    axes[0, 1].set_ylabel('Число поколений', fontsize=12)
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

    # График 4: Среднее качество популяции
    axes[1, 1].plot(dimensions_list, avg_values, 'bo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Размерность (n)', fontsize=12)
    axes[1, 1].set_ylabel('Среднее f(x)', fontsize=12)
    axes[1, 1].set_title('Среднее качество популяции', fontsize=13)
    axes[1, 1].grid(True, alpha=0.3)
    if max(avg_values) > 0:
        axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig('dimension_experiment.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results, fig


# ==================== ГЛАВНЫЕ ФУНКЦИИ ====================

def run_complete_ga_experiment():
    """Запуск полного набора экспериментов с генетическим алгоритмом."""
    print("🚀 ЗАПУСК ЛАБОРАТОРНОЙ РАБОТЫ ПО ГЕНЕТИЧЕСКОМУ АЛГОРИТМУ")
    print("=" * 70)

    all_results = {}

    # 1. Визуализация функции Растригина
    print("\n📊 1. Визуализация функции Растригина...")
    fig = visualize_rastrigin_2d()

    # 2. Базовый запуск ГА
    print("\n🧬 2. Базовый запуск генетического алгоритма...")
    base_result = genetic_algorithm(n_dim=2, **GA_BASE_PARAMS)
    print(f"   Лучшее f(x) = {base_result['best_f_value']:.6f}")
    print(f"   Поколений = {base_result['generations']}")
    print(f"   Сошелся = {base_result['converged']}")

    all_results['base_run'] = base_result

    # 3. Эксперимент с размером популяции
    print("\n📈 3. Эксперимент: Влияние размера популяции...")
    pop_results, fig_pop = experiment_population_size(n_dim=2, n_runs=3)
    all_results['population_size'] = pop_results

    # 4. Эксперимент с вероятностью мутации
    print("\n🔄 4. Эксперимент: Влияние вероятности мутации...")
    mutation_results, fig_mut = experiment_mutation_prob(n_dim=2, n_runs=3)
    all_results['mutation_prob'] = mutation_results

    # 5. Эксперимент с силой мутации
    print("\n⚡ 5. Эксперимент: Влияние силы мутации...")
    sigma_results, fig_sigma = experiment_mutation_strength(n_dim=2, n_runs=3)
    all_results['mutation_strength'] = sigma_results

    # 6. Эксперимент с размерностью
    print("\n📏 6. Эксперимент: Влияние размерности задачи...")
    dim_results, fig_dim = experiment_dimensions(n_runs=3)
    all_results['dimensions'] = dim_results

    # 7. Эксперимент с вероятностью кроссовера
    print("\n🔗 7. Эксперимент: Влияние вероятности кроссовера...")
    crossover_results, fig_cross = experiment_crossover_prob(n_dim=2, n_runs=3)
    all_results['crossover_prob'] = crossover_results

    # 8. Создаем график сходимости для базового запуска
    print("\n📈 8. Построение графиков сходимости...")
    try:
        convergence_fig = plot_convergence_curves({'Базовый ГА': base_result})
    except Exception as e:
        print(f"   ⚠️  Ошибка при построении графиков сходимости: {e}")
        print("   Пропускаем этот график...")

    print(f"\n{'=' * 70}")
    print("✅ ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА")
    print("=" * 70)
    print("\n📁 Созданные файлы:")
    print("  • rastrigin_function.png - визуализация функции")
    print("  • ga_convergence_curves.png - графики сходимости")
    print("  • Размер_популяции_N_experiment.png - влияние размера популяции")
    print("  • Вероятность_мутации_p_m_experiment.png - влияние вероятности мутации")
    print("  • Сила_мутации_σ_experiment.png - влияние силы мутации")
    print("  • Вероятность_кроссовера_p_c_experiment.png - влияние вероятности кроссовера")
    print("  • dimension_experiment.png - влияние размерности")

    return all_results


def run_quick_experiment():
    """Упрощенный запуск для быстрого тестирования."""
    print("🚀 БЫСТРЫЙ ЗАПУСК ГЕНЕТИЧЕСКОГО АЛГОРИТМА")
    print("=" * 50)

    # 1. Визуализация функции
    visualize_rastrigin_2d()

    # 2. Несколько запусков с разными параметрами
    test_configs = {
        "Базовый": GA_BASE_PARAMS,
        "Большая популяция": {**GA_BASE_PARAMS, "population_size": 100},
        "Большая мутация": {**GA_BASE_PARAMS, "mutation_prob": 0.2, "mutation_strength": 1.0},
        "Маленькая популяция": {**GA_BASE_PARAMS, "population_size": 20},
    }

    results = {}
    for name, params in test_configs.items():
        print(f"\n🧬 {name}...")
        result = genetic_algorithm(n_dim=2, **params)
        results[name] = result
        print(f"   f(x) = {result['best_f_value']:.6f}, поколений = {result['generations']}")

    # 3. График сходимости
    fig = plot_convergence_curves(results, "Сравнение разных конфигураций ГА")

    return results


def interactive_ga_experiment():
    """Интерактивный запуск ГА с пользовательскими параметрами."""
    print("\n🔧 ИНТЕРАКТИВНЫЙ ЭКСПЕРИМЕНТ С ГА")
    print("=" * 50)

    # Параметры от пользователя
    n_dim = int(input("Размерность задачи (2-20, рекомендовано 2): ") or "2")
    pop_size = int(input(f"Размер популяции (рекомендовано 50): ") or "50")
    generations = int(input(f"Макс. число поколений (рекомендовано 100): ") or "100")
    p_crossover = float(input(f"Вероятность кроссовера (0-1, рекомендовано 0.8): ") or "0.8")
    p_mutation = float(input(f"Вероятность мутации (0-1, рекомендовано 0.1): ") or "0.1")
    mutation_strength = float(input(f"Сила мутации (рекомендовано 0.5): ") or "0.5")

    # Сбор параметров
    ga_params = {
        "population_size": pop_size,
        "generations": generations,
        "crossover_prob": p_crossover,
        "mutation_prob": p_mutation,
        "mutation_strength": mutation_strength,
        "tournament_size": 3,
        "elite_count": 2,
        "alpha": 0.5,
    }

    # Запуск
    print(f"\n🧬 Запуск ГА с параметрами:")
    print(f"   Размерность: {n_dim}")
    print(f"   Популяция: {pop_size}")
    print(f"   Поколений: {generations}")
    print(f"   p_crossover: {p_crossover}")
    print(f"   p_mutation: {p_mutation}")
    print(f"   σ: {mutation_strength}")

    result = genetic_algorithm(n_dim=n_dim, **ga_params)

    # Вывод результатов
    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    print(f"   Лучшая особь: {result['best_individual']}")
    print(f"   Лучшее f(x) = {result['best_f_value']:.6f}")
    print(f"   Поколений = {result['generations']}")
    print(f"   Сошелся = {result['converged']}")
    print(f"   Время = {result['total_time']:.2f} сек")

    # График сходимости
    if n_dim == 2:
        fig = plot_convergence_curves({"Интерактивный запуск": result})

    return result


def test_specific_parameter():
    """Тестирование конкретного параметра."""
    print("\n🔍 ТЕСТИРОВАНИЕ КОНКРЕТНОГО ПАРАМЕТРА")
    print("=" * 50)
    print("Выберите параметр для тестирования:")
    print("1. Размер популяции (N)")
    print("2. Вероятность мутации (p_m)")
    print("3. Сила мутации (σ)")
    print("4. Вероятность кроссовера (p_c)")
    print("5. Размерность задачи (n)")

    param_choice = input("Введите номер (1-5): ").strip()

    n_dim = 2
    n_runs = 3

    if param_choice == "1":
        print("\n📊 Тестирование: Размер популяции")
        n_dim = int(input("Размерность задачи (по умолчанию 2): ") or "2")
        n_runs = int(input("Количество запусков (по умолчанию 3): ") or "3")
        results, fig = experiment_population_size(n_dim=n_dim, n_runs=n_runs)

    elif param_choice == "2":
        print("\n📊 Тестирование: Вероятность мутации")
        n_dim = int(input("Размерность задачи (по умолчанию 2): ") or "2")
        n_runs = int(input("Количество запусков (по умолчанию 3): ") or "3")
        results, fig = experiment_mutation_prob(n_dim=n_dim, n_runs=n_runs)

    elif param_choice == "3":
        print("\n📊 Тестирование: Сила мутации")
        n_dim = int(input("Размерность задачи (по умолчанию 2): ") or "2")
        n_runs = int(input("Количество запусков (по умолчанию 3): ") or "3")
        results, fig = experiment_mutation_strength(n_dim=n_dim, n_runs=n_runs)

    elif param_choice == "4":
        print("\n📊 Тестирование: Вероятность кроссовера")
        n_dim = int(input("Размерность задачи (по умолчанию 2): ") or "2")
        n_runs = int(input("Количество запусков (по умолчанию 3): ") or "3")
        results, fig = experiment_crossover_prob(n_dim=n_dim, n_runs=n_runs)

    elif param_choice == "5":
        print("\n📊 Тестирование: Размерность задачи")
        n_runs = int(input("Количество запусков (по умолчанию 3): ") or "3")
        results, fig = experiment_dimensions(n_runs=n_runs)

    else:
        print("❌ Неверный выбор!")
        return None

    print(f"\n✅ Тестирование завершено!")
    print(f"📁 График сохранен в файл")
    return results


# ==================== ГЛАВНОЕ МЕНЮ ====================

if __name__ == "__main__":
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА: ГЕНЕТИЧЕСКИЙ АЛГОРИТМ")
    print("=" * 60)
    print("Выберите режим работы:")
    print("1. Полный набор экспериментов (все графики + анализ)")
    print("2. Быстрый тест (несколько конфигураций)")
    print("3. Интерактивный эксперимент")
    print("4. Только визуализация функции")
    print("5. Тест конкретного параметра")

    choice = input("Введите номер (1-5): ").strip()

    if choice == "1":
        # Полный эксперимент
        all_results = run_complete_ga_experiment()

    elif choice == "2":
        # Быстрый тест
        results = run_quick_experiment()

    elif choice == "3":
        # Интерактивный
        result = interactive_ga_experiment()

    elif choice == "4":
        # Только визуализация
        print("\n📊 ВИЗУАЛИЗАЦИЯ ФУНКЦИИ РАСТРИГИНА")
        print("=" * 40)
        fig = visualize_rastrigin_2d()

    elif choice == "5":
        # Тест конкретного параметра
        results = test_specific_parameter()

    else:
        print("❌ Неверный выбор. Запускаю быстрый тест...")
        results = run_quick_experiment()

    print("\n" + "=" * 60)
    print("РАБОТА ЗАВЕРШЕНА")
    print("=" * 60)