import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.linalg import eigvalsh
import time
from tqdm import tqdm
import re

# ==================== КОНСТАНТЫ И ПАРАМЕТРЫ ДЛЯ ЭКСПЕРИМЕНТОВ ====================

# 1. РАЗНЫЕ МАТРИЦЫ A (для исследования влияния обусловленности)
MATRICES = {
    "Хорошо обусловленная (κ≈2.6)": np.array([[3, 1],
                                              [1, 2]]),

    "Плохо обусловленная (κ≈10)": np.array([[10, 0],
                                            [0, 1]]),

    "Случайная симметричная": np.random.randn(2, 2),
}

# Сделаем случайную матрицу положительно определенной
MATRICES["Случайная симметричная"] = MATRICES["Случайная симметричная"] @ MATRICES["Случайная симметричная"].T + np.eye(
    2)

# 2. НАЧАЛЬНЫЕ ТОЧКИ (разные стартовые позиции)
INITIAL_POINTS = {
    "Далекая от минимума": np.array([10.0, 10.0]),
    "Близкая к минимуму": np.array([0.5, 0.5]),
    "Случайная": np.random.randn(2) * 5,
    "Асимметричная": np.array([8.0, -5.0]),
}

# 3. ПАРАМЕТРЫ ШАГА α (для исследования скорости сходимости)
ALPHA_VALUES = {
    "Очень маленький": 0.01,
    "Маленький": 0.1,
    "Оптимальный (теоретический)": None,  # Будем вычислять для каждой матрицы
    "Близкий к пределу": None,  # Будем вычислять как 0.9 * (2/λ_max)
    "Сверх предельного": None,  # Будем вычислять как 1.1 * (2/λ_max)
}

# 4. КРИТЕРИИ ОСТАНОВКИ
STOPPING_CRITERIA = {
    "max_iterations": 1000,
    "grad_tolerance": 1e-6,
    "func_tolerance": 1e-8,
    "x_tolerance": 1e-6,
}


# ==================== ОСНОВНЫЕ ФУНКЦИИ ====================

def quadratic_function(x, A):
    """Вычисление значения квадратичной формы f(x) = 0.5 * x^T A x"""
    return 0.5 * x.T @ A @ x


def gradient(x, A):
    """Вычисление градиента ∇f(x) = A x"""
    return A @ x


def compute_optimal_alpha(A):
    """Вычисление оптимального шага α_opt = 2/(λ_min + λ_max)"""
    eigenvalues = eigvalsh(A)  # Собственные значения для симметричной матрица
    lambda_min = np.min(eigenvalues)
    lambda_max = np.max(eigenvalues)
    alpha_opt = 2.0 / (lambda_min + lambda_max)
    alpha_max = 2.0 / lambda_max  # Максимальный допустимый шаг
    return alpha_opt, alpha_max, lambda_min, lambda_max


def gradient_descent(A, x0, alpha, stopping_criteria, track_history=True):
    """
    Реализация градиентного спуска с постоянным шагом

    Parameters:
    -----------
    A : numpy.ndarray
        Симметричная положительно определенная матрица
    x0 : numpy.ndarray
        Начальная точка
    alpha : float
        Шаг обучения (learning rate)
    stopping_criteria : dict
        Критерии остановки
    track_history : bool
        Флаг сохранения истории итераций

    Returns:
    --------
    dict : Результаты оптимизации
    """
    x = x0.copy()
    f_val = quadratic_function(x, A)
    grad = gradient(x, A)
    grad_norm = np.linalg.norm(grad)

    # Инициализация истории
    history = {
        'x': [x.copy()],
        'f': [f_val],
        'grad_norm': [grad_norm],
        'time': [0.0]
    } if track_history else None

    start_time = time.time()
    iteration = 0
    converged = False

    # Основной цикл градиентного спуска
    while iteration < stopping_criteria['max_iterations']:
        # Проверка критериев остановки
        if grad_norm < stopping_criteria['grad_tolerance']:
            converged = True
            break

        if iteration > 0 and history:
            if abs(history['f'][-1] - history['f'][-2]) < stopping_criteria['func_tolerance']:
                converged = True
                break

            if np.linalg.norm(history['x'][-1] - history['x'][-2]) < stopping_criteria['x_tolerance']:
                converged = True
                break

        # Градиентный шаг
        grad = gradient(x, A)
        x = x - alpha * grad

        # Вычисление новых значений
        f_val = quadratic_function(x, A)
        grad_norm = np.linalg.norm(grad)

        # Сохранение истории
        if track_history:
            history['x'].append(x.copy())
            history['f'].append(f_val)
            history['grad_norm'].append(grad_norm)
            history['time'].append(time.time() - start_time)

        iteration += 1

    # Формирование результатов
    result = {
        'x_opt': x,
        'f_opt': f_val,
        'grad_norm_final': grad_norm,
        'iterations': iteration,
        'converged': converged,
        'history': history
    }

    return result


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def analyze_matrix(A, name):
    """Анализ матрицы: собственные значения, число обусловленности"""
    eigenvalues = eigvalsh(A)
    lambda_min, lambda_max = np.min(eigenvalues), np.max(eigenvalues)
    condition_number = lambda_max / lambda_min
    alpha_opt, alpha_max, _, _ = compute_optimal_alpha(A)

    print(f"\n{'=' * 60}")
    print(f"Анализ матрицы: {name}")
    print(f"{'=' * 60}")
    print(f"Матрица A:\n{A}")
    print(f"Собственные значения: {eigenvalues}")
    print(f"λ_min = {lambda_min:.4f}, λ_max = {lambda_max:.4f}")
    print(f"Число обусловленности κ = {condition_number:.4f}")
    print(f"Оптимальный шаг α_opt = {alpha_opt:.4f}")
    print(f"Максимальный допустимый шаг α_max = {alpha_max:.4f}")

    return {
        'lambda_min': lambda_min,
        'lambda_max': lambda_max,
        'condition_number': condition_number,
        'alpha_opt': alpha_opt,
        'alpha_max': alpha_max
    }


def plot_convergence(results_dict, title="Сравнение сходимости"):
    """Построение графиков сходимости для разных параметров"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(results_dict)))

    for (label, result), color in zip(results_dict.items(), colors):
        if result['history']:
            history = result['history']
            iterations = list(range(len(history['f'])))

            # График 1: Значение функции
            axes[0, 0].semilogy(iterations, history['f'],
                                label=label, color=color, linewidth=2)
            axes[0, 0].set_xlabel('Итерация')
            axes[0, 0].set_ylabel('f(x) (log scale)')
            axes[0, 0].set_title('Сходимость функции')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()

            # График 2: Норма градиента
            axes[0, 1].semilogy(iterations, history['grad_norm'],
                                label=label, color=color, linewidth=2)
            axes[0, 1].set_xlabel('Итерация')
            axes[0, 1].set_ylabel('||∇f(x)|| (log scale)')
            axes[0, 1].set_title('Убывание нормы градиента')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()

            # График 3: Траектория в пространстве параметров (для 2D)
            if len(result['x_opt']) == 2:
                x_vals = [x[0] for x in history['x']]
                y_vals = [x[1] for x in history['x']]
                axes[1, 0].plot(x_vals, y_vals, 'o-',
                                label=label, color=color,
                                markersize=4, linewidth=1.5, alpha=0.7)
                axes[1, 0].plot(x_vals[0], y_vals[0], 'go', markersize=10, label='Старт')
                axes[1, 0].plot(x_vals[-1], y_vals[-1], 'r*', markersize=15, label='Финиш')
                axes[1, 0].set_xlabel('x₁')
                axes[1, 0].set_ylabel('x₂')
                axes[1, 0].set_title('Траектория оптимизации')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
                axes[1, 0].axis('equal')

            # График 4: Время выполнения
            axes[1, 1].plot(history['time'], history['f'],
                            label=label, color=color, linewidth=2)
            axes[1, 1].set_xlabel('Время (сек)')
            axes[1, 1].set_ylabel('f(x)')
            axes[1, 1].set_title('Сходимость по времени')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()

    plt.tight_layout()
    return fig


def plot_contour_with_trajectories(A, results_dict, x_range=(-2, 12), y_range=(-2, 12)):
    """Визуализация линий уровня и траекторий для 2D случая"""
    # Создание сетки для линий уровня
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)

    # Вычисление значений функции на сетке
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = quadratic_function(point, A)

    # Построение контурного графика
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.7)
    ax.clabel(contour, inline=True, fontsize=8)

    # Отображение траекторий
    colors = plt.cm.rainbow(np.linspace(0, 1, len(results_dict)))

    for (label, result), color in zip(results_dict.items(), colors):
        if result['history'] and len(result['x_opt']) == 2:
            history = result['history']
            x_vals = [x[0] for x in history['x']]
            y_vals = [x[1] for x in history['x']]

            ax.plot(x_vals, y_vals, 'o-', color=color,
                    linewidth=2, markersize=4, alpha=0.8, label=label)
            ax.plot(x_vals[0], y_vals[0], 'o', color=color,
                    markersize=10, markeredgecolor='black', markeredgewidth=2)
            ax.plot(x_vals[-1], y_vals[-1], '*', color=color,
                    markersize=15, markeredgecolor='black', markeredgewidth=1)

    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Линии уровня и траектории градиентного спуска', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')

    return fig


# ==================== РАЗДЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ ТРАЕКТОРИЙ СПУСКА ====================

def plot_contour_separate_axes(A, results_dict, x_range=(-2, 12), y_range=(-2, 12)):
    """
    Визуализация линий уровня на РАЗНЫХ координатных плоскостях
    Каждая траектория на своем отдельном графике
    """
    # Создание сетки для линий уровня
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)

    # Вычисление значений функции на сетке
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = quadratic_function(point, A)

    # Определяем сколько графиков нужно
    n_results = len(results_dict)
    n_cols = min(3, n_results)  # максимум 3 колонки
    n_rows = (n_results + n_cols - 1) // n_cols

    # Увеличиваем размер фигуры для лучшего отображения заголовков
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    # Если только один график, axes - не массив
    if n_results == 1:
        axes = np.array([axes])
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    colors = plt.cm.tab10(np.linspace(0, 1, n_results))

    for idx, ((label, result), color) in enumerate(zip(results_dict.items(), colors)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Построение линий уровня
        levels = np.linspace(np.min(Z), np.max(Z), 20)
        contour = ax.contour(X, Y, Z, levels=levels, cmap='Blues', alpha=0.6, linewidths=0.8)
        ax.clabel(contour, inline=True, fontsize=7, fmt='%1.1f')

        # Отображение траектории
        if result['history'] and len(result['x_opt']) == 2:
            history = result['history']
            x_vals = np.array([x[0] for x in history['x']])
            y_vals = np.array([x[1] for x in history['x']])

            # Рисуем траекторию
            ax.plot(x_vals, y_vals, 'o-', color=color,
                    linewidth=2, markersize=4, alpha=0.9,
                    markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5)

            # Начальная точка
            ax.plot(x_vals[0], y_vals[0], 'o', color='green',
                    markersize=8, markeredgecolor='black', markeredgewidth=1.5,
                    label='Старт')

            # Конечная точка
            ax.plot(x_vals[-1], y_vals[-1], 's', color='red',
                    markersize=8, markeredgecolor='black', markeredgewidth=1.5,
                    label='Финиш')

        # Настройки графика - УПРОЩЕННЫЙ заголовок
        ax.set_xlabel('x₁', fontsize=10)
        ax.set_ylabel('x₂', fontsize=10)

        # Создаем КОРОТКИЙ информативный заголовок
        if result['converged']:
            status = "✓"
        else:
            status = "✗"

        # Берем только ключевую часть названия
        short_label = label.split()[0] if ' ' in label else label[:15]
        ax.set_title(f'{short_label} {status} ({result["iterations"]} ит.)',
                     fontsize=11, fontweight='bold', pad=10)  # pad добавляет отступ

        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(loc='upper right', fontsize=8)
        ax.axis('equal')

    # Скрываем пустые графики
    for idx in range(n_results, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    # Общий заголовок - ПЕРЕНОСИМ его в подпись
    eigenvalues = eigvalsh(A)
    lambda_min, lambda_max = np.min(eigenvalues), np.max(eigenvalues)
    condition_number = lambda_max / lambda_min

    # Вместо fig.suptitle используем text для более гибкого размещения
    info_text = (
        f'Матрица A = [[{A[0, 0]:.1f}, {A[0, 1]:.1f}], [{A[1, 0]:.1f}, {A[1, 1]:.1f}]]\n'
        f'Число обусловленности κ = {condition_number:.2f}, '
        f'Диапазон: ({x_range[0]}, {y_range[0]}) → ({x_range[1]}, {y_range[1]})'
    )

    # Добавляем текст внизу
    plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Главный заголовок - компактный
    fig.suptitle(
        'Градиентный спуск: траектории на отдельных осях',
        fontsize=13, fontweight='bold', y=0.98  # y=0.98 - опускаем чуть ниже
    )

    # Увеличиваем отступы
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # rect: [left, bottom, right, top]
    return fig


def plot_comparison_grid(A, results_dict, x_range=(-2, 12), y_range=(-2, 12)):
    """
    Сетка сравнения: слева - все вместе, справа - отдельные
    """
    # Создание сетки для линий уровня
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)

    # Вычисление значений функции на сетке
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = quadratic_function(point, A)

    # Увеличиваем размер фигуры
    fig = plt.figure(figsize=(18, 9))

    # 1. Все траектории вместе
    ax1 = plt.subplot(1, 2, 1)

    levels = np.linspace(np.min(Z), np.max(Z), 20)
    contour = ax1.contour(X, Y, Z, levels=levels, cmap='Blues', alpha=0.6, linewidths=0.8)
    ax1.clabel(contour, inline=True, fontsize=7, fmt='%1.1f')

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

    for (label, result), color in zip(results_dict.items(), colors):
        if result['history'] and len(result['x_opt']) == 2:
            history = result['history']
            x_vals = np.array([x[0] for x in history['x']])
            y_vals = np.array([x[1] for x in history['x']])

            # Создаем короткую метку для легенды
            short_label = label.split()[0] if ' ' in label else label[:15]
            ax1.plot(x_vals, y_vals, 'o-', color=color,
                     linewidth=1.5, markersize=3, alpha=0.7,
                     label=short_label)

    ax1.set_xlabel('x₁', fontsize=11)
    ax1.set_ylabel('x₂', fontsize=11)
    ax1.set_title('Все траектории вместе', fontsize=12, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.axis('equal')

    # 2. Отдельные траектории (первые 4) - увеличиваем оси
    gs = plt.GridSpec(2, 4, figure=fig, wspace=0.3, hspace=0.4)

    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])
    ax4 = fig.add_subplot(gs[1, 2])
    ax5 = fig.add_subplot(gs[1, 3])

    separate_axes = [ax2, ax3, ax4, ax5]

    for idx, ((label, result), color) in enumerate(zip(results_dict.items(), colors)):
        if idx >= 4:  # Показываем только первые 4
            break

        ax = separate_axes[idx]

        # Линии уровня
        contour_single = ax.contour(X, Y, Z, levels=levels, cmap='Blues', alpha=0.5, linewidths=0.6)

        if result['history'] and len(result['x_opt']) == 2:
            history = result['history']
            x_vals = np.array([x[0] for x in history['x']])
            y_vals = np.array([x[1] for x in history['x']])

            ax.plot(x_vals, y_vals, 'o-', color=color,
                    linewidth=2, markersize=4, alpha=0.9)
            ax.plot(x_vals[0], y_vals[0], 'go', markersize=6, label='Старт')
            ax.plot(x_vals[-1], y_vals[-1], 'rs', markersize=6, label='Финиш')

        # Упрощенный заголовок
        short_title = label.split()[0] if ' ' in label else label[:10]
        status = "✓" if result['converged'] else "✗"
        ax.set_title(f'{short_title} {status}', fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('x₁', fontsize=9)
        ax.set_ylabel('x₂', fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.axis('equal')
        ax.legend(loc='upper right', fontsize=8)

    # Общая информация - внизу
    eigenvalues = eigvalsh(A)
    lambda_min, lambda_max = np.min(eigenvalues), np.max(eigenvalues)
    condition_number = lambda_max / lambda_min

    info_text = (
        f'Матрица A = [[{A[0, 0]:.1f}, {A[0, 1]:.1f}], [{A[1, 0]:.1f}, {A[1, 1]:.1f}]], '
        f'κ = {condition_number:.2f}\n'
        f'Диапазон: x₁ ∈ [{x_range[0]}, {x_range[1]}], x₂ ∈ [{y_range[0]}, {y_range[1]}]'
    )

    plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Главный заголовок
    fig.suptitle(
        'Сравнение траекторий градиентного спуска',
        fontsize=14, fontweight='bold', y=0.97
    )

    # Регулируем отступы
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    return fig


def save_clean_separate_plots(A, results_dict, x_range=(-2, 12), y_range=(-2, 12),
                              filename='gradient_trajectories_clean.png'):
    """
    Сохраняет чистые графики без обрезанных заголовков
    """
    # Создание сетки для линий уровня
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)

    # Вычисление значений функции на сетке
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = quadratic_function(point, A)

    # Создаем отдельную фигуру для каждого результата
    for label, result in results_dict.items():
        if result['history'] and len(result['x_opt']) == 2:
            # Новая фигура для каждого графика
            fig, ax = plt.subplots(figsize=(8, 7))

            # Построение линий уровня
            levels = np.linspace(np.min(Z), np.max(Z), 20)
            contour = ax.contour(X, Y, Z, levels=levels, cmap='Blues', alpha=0.6, linewidths=0.8)
            ax.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')

            # Траектория
            history = result['history']
            x_vals = np.array([x[0] for x in history['x']])
            y_vals = np.array([x[1] for x in history['x']])

            ax.plot(x_vals, y_vals, 'o-', color='red',
                    linewidth=2, markersize=5, alpha=0.9,
                    markerfacecolor='white', markeredgecolor='red', markeredgewidth=1.5)

            ax.plot(x_vals[0], y_vals[0], 'go', markersize=10,
                    markeredgecolor='black', markeredgewidth=2, label='Старт')
            ax.plot(x_vals[-1], y_vals[-1], 'rs', markersize=10,
                    markeredgecolor='black', markeredgewidth=2, label='Финиш')

            # Информация о матрице
            eigenvalues = eigvalsh(A)
            lambda_min, lambda_max = np.min(eigenvalues), np.max(eigenvalues)
            condition_number = lambda_max / lambda_min

            # Полный заголовок
            title_text = (
                f'Градиентный спуск: {label}\n'
                f'Матрица A = [[{A[0, 0]:.1f}, {A[0, 1]:.1f}], [{A[1, 0]:.1f}, {A[1, 1]:.1f}]], '
                f'κ = {condition_number:.2f}\n'
                f'Итераций: {result["iterations"]}, '
                f'f(x*) = {result["f_opt"]:.2e}'
            )

            ax.set_title(title_text, fontsize=12, fontweight='bold', pad=15)
            ax.set_xlabel('x₁', fontsize=11)
            ax.set_ylabel('x₂', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.legend(loc='upper right', fontsize=10)
            ax.axis('equal')

            # Сохраняем каждый график отдельно
            safe_label = label.replace('/', '_').replace('\\', '_').replace(':', '_')
            plt.tight_layout()
            plt.savefig(f'trajectory_{safe_label}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

    print(f"\n✅ Отдельные графики сохранены в файлы trajectory_*.png")


# ==================== ФУНКЦИИ ДЛЯ АНАЛИЗА РЕЗУЛЬТАТОВ ====================

def print_detailed_analysis(results_alpha, results_condition, A_original=None):
    """Расширенный анализ результатов всех экспериментов"""
    print("\n" + "=" * 100)
    print("РАСШИРЕННЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТОВ")
    print("=" * 100)

    # 1. Сводная таблица для α экспериментов
    print("\n1. СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (ВЛИЯНИЕ α):")
    print("-" * 90)
    print(f"{'Эксперимент':<30} {'α':<10} {'Итерации':<12} {'f(x*)':<15} {'||∇f||':<15} {'Время (с)':<12}")
    print("-" * 90)

    for label, result in results_alpha.items():
        if result['history']:
            time_total = result['history']['time'][-1] if result['history']['time'] else 0
        else:
            time_total = 0

        # Извлекаем α из названия или используем значение
        alpha_value = extract_alpha_from_label(label, A_original)
        print(f"{label:<30} {alpha_value:<10.4f} {result['iterations']:<12} "
              f"{result['f_opt']:<15.6e} {result['grad_norm_final']:<15.6e} {time_total:<12.6f}")

    print("-" * 90)

    # 2. Теоретический анализ
    if A_original is not None:
        print("\n2. ТЕОРЕТИЧЕСКИЙ АНАЛИЗ МАТРИЦЫ A:")
        print("-" * 70)

        eigenvalues = eigvalsh(A_original)
        lambda_min, lambda_max = np.min(eigenvalues), np.max(eigenvalues)
        condition_number = lambda_max / lambda_min
        alpha_opt, alpha_max, _, _ = compute_optimal_alpha(A_original)

        print(f"Собственные значения: λ_min = {lambda_min:.4f}, λ_max = {lambda_max:.4f}")
        print(f"Число обусловленности: κ = {condition_number:.4f}")
        print(f"Теоретический диапазон α: 0 < α < {alpha_max:.4f}")
        print(f"Оптимальный теоретический α: α_opt = {alpha_opt:.4f}")
        print(f"Скорость сходимости (теоретическая): q = {(condition_number - 1) / (condition_number + 1):.4f}")

        # Собственные векторы
        eigenvectors = np.linalg.eigh(A_original)[1]
        print(f"\nСобственные векторы:")
        print(f"v₁ (соответствует λ_min = {lambda_min:.4f}): {eigenvectors[:, 0]}")
        print(f"v₂ (соответствует λ_max = {lambda_max:.4f}): {eigenvectors[:, 1]}")

    # 3. Анализ влияния обусловленности
    if results_condition:
        print("\n3. АНАЛИЗ ВЛИЯНИЯ ЧИСЛА ОБУСЛОВЛЕННОСТИ κ:")
        print("-" * 70)
        print(f"{'Матрица':<30} {'κ':<10} {'Итерации':<12} {'Скорость (1/ит.)':<15}")
        print("-" * 70)

        for label, result in results_condition.items():
            # Извлекаем κ из названия или вычисляем
            if "κ" in label:
                kappa_match = re.search(r'κ[≈=]?([\d.]+)', label)
                kappa = float(kappa_match.group(1)) if kappa_match else 1
            else:
                kappa = 1

            speed = 1 / result['iterations'] if result['iterations'] > 0 else 0
            print(f"{label:<30} {kappa:<10.2f} {result['iterations']:<12} {speed:<15.4f}")

    # 4. Качественный анализ
    print("\n4. КАЧЕСТВЕННЫЙ АНАЛИЗ:")
    print("-" * 70)

    # Анализ поведения при разных α
    print("\n4.1. Сравнение теоретических и практических границ α:")
    if A_original is not None:
        for label, result in results_alpha.items():
            alpha_value = extract_alpha_from_label(label, A_original)
            if alpha_value > alpha_max:
                print(f"  • {label}: α={alpha_value:.4f} > α_max={alpha_max:.4f} → РАСХОДИМОСТЬ")
            elif alpha_value > 0.9 * alpha_max:
                print(f"  • {label}: α={alpha_value:.4f} близко к α_max={alpha_max:.4f} → КОЛЕБАНИЯ")
            elif abs(alpha_value - alpha_opt) < 0.1 * alpha_opt:
                print(f"  • {label}: α={alpha_value:.4f} ≈ α_opt={alpha_opt:.4f} → ОПТИМАЛЬНО")
            elif alpha_value < 0.1 * alpha_opt:
                print(f"  • {label}: α={alpha_value:.4f} << α_opt={alpha_opt:.4f} → МЕДЛЕННО")

    print("\n4.2. Объяснение формы траекторий через собственные векторы:")
    if A_original is not None:
        eigenvectors = np.linalg.eigh(A_original)[1]
        print(f"  • Собственный вектор v₁ (λ_min={lambda_min:.4f}): {eigenvectors[:, 0]}")
        print(f"    Направление наиболее пологого спуска")
        print(f"  • Собственный вектор v₂ (λ_max={lambda_max:.4f}): {eigenvectors[:, 1]}")
        print(f"    Направление наиболее крутого спуска")
        print(f"  • Зигзагообразные траектории возникают из-за разной скорости движения")
        print(f"    вдоль этих направлений")

    print("\n4.3. Анализ причин замедления сходимости при плохой обусловленности:")
    print("  • Большое κ означает, что линии уровня сильно вытянуты (эллипсы)")
    print("  • Градиент указывает почти перпендикулярно к направлению к минимуму")
    print("  • Алгоритм делает много зигзагов, тратя время на коррекцию направления")

    print("\n4.4. Рекомендации по выбору α для реальных задач:")
    print("  • Начинать с α ≈ 0.1 * α_max (осторожный подход)")
    print("  • Использовать адаптивные методы выбора шага")
    print("  • Для плохо обусловленных задач применять методы с моментом")
    print("  • Мониторить норму градиента для ранней остановки")

    # 5. Выводы
    print("\n5. ВЫВОДЫ ПО РАЗДЕЛУ:")
    print("-" * 70)

    # Находим оптимальный α по результатам
    best_alpha = None
    best_iterations = float('inf')
    for label, result in results_alpha.items():
        if result['converged'] and result['iterations'] < best_iterations:
            best_iterations = result['iterations']
            best_alpha = extract_alpha_from_label(label, A_original)

    print("5.1. Критическая важность выбора правильного шага α:")
    print(f"  • Оптимальный шаг в эксперименте: α ≈ {best_alpha:.4f}")
    print(f"  • Разница в итерациях между лучшим и худшим случаем: ", end="")

    iterations = [r['iterations'] for r in results_alpha.values() if r['converged']]
    if iterations:
        print(f"{max(iterations) - min(iterations)} итераций ({max(iterations) / min(iterations):.1f}×)")

    print("\n5.2. Прямая зависимость скорости сходимости от κ:")
    if results_condition:
        kappas = []
        speeds = []
        for label, result in results_condition.items():
            if "κ" in label:
                kappa_match = re.search(r'κ[≈=]?([\d.]+)', label)
                if kappa_match:
                    kappas.append(float(kappa_match.group(1)))
                    speeds.append(1 / result['iterations'] if result['iterations'] > 0 else 0)

        if len(kappas) >= 2:
            print(f"  • При увеличении κ с {min(kappas):.1f} до {max(kappas):.1f}")
            print(f"    скорость уменьшилась в {max(speeds) / min(speeds):.1f} раз")

    print("\n5.3. Наглядная геометрическая интерпретация метода:")
    print("  • Линии уровня показывают 'ландшафт' функции")
    print("  • Траектории демонстрируют путь градиентного спуска")
    print("  • Направление градиента всегда перпендикулярно линии уровня")

    print("\n5.4. Ограничения базового градиентного спуска:")
    print("  • Чувствительность к выбору шага α")
    print("  • Медленная сходимость при плохой обусловленности")
    print("  • Требует вычисления градиента на каждой итерации")
    print("\n  Направления для улучшения:")
    print("  • Градиентный спуск с моментом (Momentum)")
    print("  • Адаптивные методы (Adam, RMSprop)")
    print("  • Методы второго порядка (Ньютона)")

    print("\n" + "=" * 100)


def extract_alpha_from_label(label, A=None):
    """Извлекает значение α из названия эксперимента"""
    # Пытаемся найти α в названии
    alpha_match = re.search(r'α[=: ]*([\d.]+)', label)
    if alpha_match:
        return float(alpha_match.group(1))

    # Если не нашли, используем стандартные значения
    if "Очень маленький" in label:
        return 0.01
    elif "Маленький" in label:
        return 0.1
    elif "Оптимальный" in label:
        if A is not None:
            alpha_opt, _, _, _ = compute_optimal_alpha(A)
            return alpha_opt
        return 0.4
    elif "Близкий к пределу" in label:
        if A is not None:
            _, alpha_max, _, _ = compute_optimal_alpha(A)
            return 0.9 * alpha_max
        return 0.52
    elif "Сверх предельного" in label:
        if A is not None:
            _, alpha_max, _, _ = compute_optimal_alpha(A)
            return 1.1 * alpha_max
        return 0.6
    else:
        return 0.1  # значение по умолчанию


def calculate_convergence_rate(results_dict, A=None):
    """Вычисляет скорость сходимости для каждого эксперимента"""
    print("\n" + "=" * 80)
    print("АНАЛИЗ СКОРОСТИ СХОДИМОСТИ")
    print("=" * 80)

    convergence_rates = {}

    for label, result in results_dict.items():
        if result['history'] and len(result['history']['f']) > 10:
            f_values = result['history']['f']

            # Вычисляем среднюю скорость убывания
            rates = []
            for i in range(1, len(f_values)):
                if f_values[i - 1] > 0:
                    rate = f_values[i] / f_values[i - 1]
                    rates.append(rate)

            if rates:
                avg_rate = np.mean(rates)
                convergence_rates[label] = avg_rate

                # Теоретическая скорость
                theoretical_rate = None
                if A is not None:
                    alpha_value = extract_alpha_from_label(label, A)
                    eigenvalues = eigvalsh(A)
                    lambda_min, lambda_max = np.min(eigenvalues), np.max(eigenvalues)
                    theoretical_rate = max(abs(1 - alpha_value * lambda_min),
                                           abs(1 - alpha_value * lambda_max))

                print(f"{label:<30}: Средняя скорость = {avg_rate:.4f}", end="")
                if theoretical_rate:
                    print(f" (теоретическая: {theoretical_rate:.4f})")
                else:
                    print()

    return convergence_rates


def plot_convergence_summary(results_alpha, results_condition):
    """Строит сводные графики для анализа"""
    # Создаем фигуру с несколькими графиками
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Сводный анализ результатов экспериментов', fontsize=16, fontweight='bold')

    # График 1: Итерации vs α
    ax1 = axes[0, 0]
    alphas = []
    iterations = []
    labels = []

    for label, result in results_alpha.items():
        alpha_val = extract_alpha_from_label(label)
        alphas.append(alpha_val)
        iterations.append(result['iterations'])
        labels.append(label.split()[0])

    ax1.scatter(alphas, iterations, c='red', s=100, alpha=0.7)
    for i, (alpha, iter_count, label) in enumerate(zip(alphas, iterations, labels)):
        ax1.annotate(label, (alpha, iter_count), xytext=(5, 5),
                     textcoords='offset points', fontsize=9)

    ax1.set_xlabel('Шаг α', fontsize=12)
    ax1.set_ylabel('Число итераций', fontsize=12)
    ax1.set_title('Зависимость числа итераций от α', fontsize=13)
    ax1.grid(True, alpha=0.3)

    # График 2: Скорость сходимости
    ax2 = axes[0, 1]
    if results_condition:
        kappas = []
        speeds = []
        cond_labels = []

        for label, result in results_condition.items():
            if "κ" in label:
                kappa_match = re.search(r'κ[≈=]?([\d.]+)', label)
                if kappa_match:
                    kappa = float(kappa_match.group(1))
                    kappas.append(kappa)
                    speeds.append(1 / result['iterations'] if result['iterations'] > 0 else 0)
                    cond_labels.append(label.split()[0])

        if kappas:
            ax2.plot(kappas, speeds, 'bo-', linewidth=2, markersize=8)
            ax2.set_xlabel('Число обусловленности κ', fontsize=12)
            ax2.set_ylabel('Скорость сходимости (1/итерации)', fontsize=12)
            ax2.set_title('Влияние κ на скорость сходимости', fontsize=13)
            ax2.grid(True, alpha=0.3)

    # График 3: Сравнение эффективности разных α
    ax3 = axes[1, 0]
    efficiency = []
    alpha_labels = []

    for label, result in results_alpha.items():
        if result['history']:
            time_total = result['history']['time'][-1] if result['history']['time'] else 0
            # Эффективность = 1/(итерации * время)
            if result['iterations'] > 0 and time_total > 0:
                efficiency.append(1 / (result['iterations'] * time_total))
            else:
                efficiency.append(0)
            alpha_labels.append(label.split()[0])

    bars = ax3.bar(range(len(efficiency)), efficiency, color=plt.cm.viridis(np.linspace(0, 1, len(efficiency))))
    ax3.set_xticks(range(len(efficiency)))
    ax3.set_xticklabels(alpha_labels, rotation=45, ha='right')
    ax3.set_ylabel('Эффективность (1/(ит.×время))', fontsize=12)
    ax3.set_title('Сравнение эффективности разных α', fontsize=13)

    # Подписываем значения на столбцах
    for bar, eff in zip(bars, efficiency):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                 f'{eff:.4f}', ha='center', va='bottom', fontsize=9)

    # График 4: Градиент по итерациям для лучшего и худшего случая
    ax4 = axes[1, 1]

    # Находим лучший и худший случаи по числу итераций
    converged_results = {k: v for k, v in results_alpha.items() if v['converged']}
    if len(converged_results) >= 2:
        best_label = min(converged_results, key=lambda k: converged_results[k]['iterations'])
        worst_label = max(converged_results, key=lambda k: converged_results[k]['iterations'])

        best_result = converged_results[best_label]
        worst_result = converged_results[worst_label]

        if best_result['history'] and worst_result['history']:
            best_grad = best_result['history']['grad_norm'][:100]  # первые 100 итераций
            worst_grad = worst_result['history']['grad_norm'][:100]

            ax4.semilogy(range(len(best_grad)), best_grad, 'g-', linewidth=2, label=f'Лучший: {best_label}')
            ax4.semilogy(range(len(worst_grad)), worst_grad, 'r-', linewidth=2, label=f'Худший: {worst_label}')

            ax4.set_xlabel('Итерация', fontsize=12)
            ax4.set_ylabel('||∇f|| (log scale)', fontsize=12)
            ax4.set_title('Сравнение убывания градиента', fontsize=13)
            ax4.grid(True, alpha=0.3)
            ax4.legend()

    plt.tight_layout()
    return fig


def create_results_table(results_dict):
    """Создание таблицы с результатами экспериментов"""
    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)
    print(f"{'Эксперимент':<30} {'Итерации':<10} {'f(x*)':<15} {'||∇f||':<15} {'Сходимость':<12}")
    print("-" * 80)

    for label, result in results_dict.items():
        print(f"{label:<30} {result['iterations']:<10} {result['f_opt']:<15.6e} "
              f"{result['grad_norm_final']:<15.6e} {str(result['converged']):<12}")

    print("=" * 80)


# ==================== ОСНОВНОЙ ЭКСПЕРИМЕНТ С ДОПОЛНЕНИЕМ ====================

def run_complete_experiment():
    """Запуск полного набора экспериментов - ОБНОВЛЕННАЯ ВЕРСИЯ"""
    print("🚀 ЗАПУСК ЛАБОРАТОРНОЙ РАБОТЫ ПО ГРАДИЕНТНОМУ СПУСКУ")
    print("=" * 70)

    # Выбираем матрицу для основного эксперимента
    A = MATRICES["Хорошо обусловленная (κ≈2.6)"]
    matrix_info = analyze_matrix(A, "Основная матрица")

    # Начальная точка
    x0 = INITIAL_POINTS["Далекая от минимума"]
    print(f"\nНачальная точка: x0 = {x0}")
    print(f"Начальное значение функции: f(x0) = {quadratic_function(x0, A):.4f}")

    # Вычисляем параметры шага для этой матрицы
    alpha_opt, alpha_max, _, _ = compute_optimal_alpha(A)

    # Заполняем значения шагов
    ALPHA_VALUES["Оптимальный (теоретический)"] = alpha_opt
    ALPHA_VALUES["Близкий к пределу"] = 0.9 * alpha_max
    ALPHA_VALUES["Сверх предельного"] = 1.1 * alpha_max

    print(f"\nПараметры шага:")
    for name, value in ALPHA_VALUES.items():
        print(f"  {name}: {value:.4f}")

    # Запуск экспериментов с разными шагами
    results_alpha = {}

    print(f"\n{'=' * 60}")
    print("ЭКСПЕРИМЕНТ 1: Влияние размера шага α на сходимость")
    print(f"{'=' * 60}")

    for alpha_name, alpha_value in tqdm(ALPHA_VALUES.items(), desc="Запуск экспериментов"):
        result = gradient_descent(A, x0, alpha_value, STOPPING_CRITERIA)
        results_alpha[alpha_name] = result

        print(f"\n{alpha_name} (α={alpha_value:.4f}):")
        print(f"  Итерации: {result['iterations']}")
        print(f"  f(x*) = {result['f_opt']:.6e}")
        print(f"  ||∇f(x*)|| = {result['grad_norm_final']:.6e}")
        print(f"  Сошелся: {result['converged']}")

    # ВИЗУАЛИЗАЦИЯ - ТЕПЕРЬ С РАЗНЫМИ ВАРИАНТАМИ
    print("\n📊 Построение графиков...")

    # 1. Графики сходимости
    fig1 = plot_convergence(results_alpha,
                            "Влияние размера шага α на сходимость градиентного спуска")

    # 2. Линии уровня с траекториями ВСЕ ВМЕСТЕ
    fig2 = plot_contour_with_trajectories(A, results_alpha, x_range=(-1, 11), y_range=(-1, 11))

    # 3. Линии уровень на ОТДЕЛЬНЫХ осях (исправленная версия)
    fig3 = plot_contour_separate_axes(A, results_alpha, x_range=(-1, 11), y_range=(-1, 11))

    # 4. Сетка сравнения (исправленная версия)
    fig4 = plot_comparison_grid(A, results_alpha, x_range=(-1, 11), y_range=(-1, 11))

    # 5. Отдельные чистые графики (дополнительно)
    save_clean_separate_plots(A, results_alpha, x_range=(-1, 11), y_range=(-1, 11))

    # 6. Анализ скорости сходимости
    convergence_rates = calculate_convergence_rate(results_alpha, A)

    # ЭКСПЕРИМЕНТ 2: Влияние начальной точки
    print(f"\n{'=' * 60}")
    print("ЭКСПЕРИМЕНТ 2: Влияние начальной точки")
    print(f"{'=' * 60}")

    alpha_fixed = alpha_opt
    results_points = {}

    for point_name, point_value in INITIAL_POINTS.items():
        result = gradient_descent(A, point_value, alpha_fixed, STOPPING_CRITERIA)
        results_points[point_name] = result

        print(f"\n{point_name}: x0 = {point_value}")
        print(f"  Итерации: {result['iterations']}")
        print(f"  f(x*) = {result['f_opt']:.6e}")
        print(f"  Сошелся: {result['converged']}")

    # ЭКСПЕРИМЕНТ 3: Влияние числа обусловленности
    print(f"\n{'=' * 60}")
    print("ЭКСПЕРИМЕНТ 3: Влияние числа обусловленности κ")
    print(f"{'=' * 60}")

    results_condition = {}
    x0_fixed = np.array([5.0, 5.0])

    for matrix_name, matrix_A in MATRICES.items():
        if matrix_name != "Случайная симметричная":  # Пропускаем случайную для чистоты эксперимента
            # Анализ матрицы
            info = analyze_matrix(matrix_A, matrix_name)

            # Используем оптимальный шаг для каждой матрицы
            alpha_for_matrix = info['alpha_opt']

            # Запуск градиентного спуска
            result = gradient_descent(matrix_A, x0_fixed, alpha_for_matrix, STOPPING_CRITERIA)
            results_condition[matrix_name] = result

            print(f"\n{matrix_name} (κ={info['condition_number']:.2f}):")
            print(f"  Итерации: {result['iterations']}")
            print(f"  f(x*) = {result['f_opt']:.6e}")
            print(f"  Сошелся: {result['converged']}")

    # 7. Детальный анализ результатов
    print_detailed_analysis(results_alpha, results_condition, A)

    # 8. Сводные графики анализа
    fig5 = plot_convergence_summary(results_alpha, results_condition)
    fig5.savefig('convergence_summary.png', dpi=300, bbox_inches='tight')

    # Визуализация для разных матриц
    if len(results_condition) > 0:
        fig6 = plot_convergence(results_condition,
                                "Влияние числа обусловленности на сходимость")
        fig6.savefig('condition_number_convergence.png', dpi=300, bbox_inches='tight')

    # Сохранение графиков
    fig1.savefig('gradient_descent_convergence.png', dpi=300, bbox_inches='tight')
    fig2.savefig('gradient_descent_contours_combined.png', dpi=300, bbox_inches='tight')
    fig3.savefig('gradient_descent_contours_separate.png', dpi=300, bbox_inches='tight')
    fig4.savefig('gradient_descent_comparison_grid.png', dpi=300, bbox_inches='tight')

    print(f"\n✅ Эксперименты завершены!")
    print(f"📁 Графики сохранены:")
    print(f"  • gradient_descent_convergence.png - графики сходимости")
    print(f"  • gradient_descent_contours_combined.png - все траектории вместе")
    print(f"  • gradient_descent_contours_separate.png - траектории на отдельных осях")
    print(f"  • gradient_descent_comparison_grid.png - сетка сравнения")
    print(f"  • convergence_summary.png - сводный анализ")

    return results_alpha, results_points, results_condition


def run_simple_experiment_with_separate_plots():
    """Упрощенный запуск только с отдельными графиками"""
    print("🚀 ЗАПУСК УПРОЩЕННОГО ЭКСПЕРИМЕНТА")
    print("=" * 50)

    A = MATRICES["Хорошо обусловленная (κ≈2.6)"]
    x0 = INITIAL_POINTS["Далекая от минимума"]

    print(f"Матрица A:\n{A}")
    print(f"Начальная точка: x0 = {x0}")

    # Тестируем разные α
    test_alphas = {
        "α = 0.05 (очень маленький)": 0.05,
        "α = 0.1 (маленький)": 0.1,
        "α = 0.4 (оптимальный)": 0.4,
        "α = 0.52 (близкий к пределу)": 0.52,
        "α = 0.6 (сверх предела)": 0.6
    }

    results = {}
    print("\nЗапуск градиентного спуска...")
    for label, alpha in test_alphas.items():
        result = gradient_descent(A, x0, alpha, STOPPING_CRITERIA)
        results[label] = result
        status = "✓" if result['converged'] else "✗"
        print(f"{status} {label}: итераций={result['iterations']}")

    # Строим отдельные графики
    fig1 = plot_contour_separate_axes(A, results, x_range=(-1, 11), y_range=(-1, 11))
    fig1.savefig('separate_axes_simple.png', dpi=300, bbox_inches='tight')

    # Сетку сравнения
    fig2 = plot_comparison_grid(A, results, x_range=(-1, 11), y_range=(-1, 11))
    fig2.savefig('comparison_grid_simple.png', dpi=300, bbox_inches='tight')

    print("\n✅ Графики сохранены:")
    print(f"  • separate_axes_simple.png - отдельные оси")
    print(f"  • comparison_grid_simple.png - сетка сравнения")

    return results


# ==================== ФУНКЦИИ ДЛЯ ИНТЕРАКТИВНОГО ИССЛЕДОВАНИЯ ====================

def interactive_experiment(A=None, x0=None, alpha=None):
    """Интерактивный запуск одного эксперимента"""
    if A is None:
        A = MATRICES["Хорошо обусловленная (κ≈2.6)"]

    if x0 is None:
        x0 = INITIAL_POINTS["Далекая от минимума"]

    if alpha is None:
        alpha, _, _, _ = compute_optimal_alpha(A)

    print(f"\n🔍 ИНТЕРАКТИВНЫЙ ЭКСПЕРИМЕНТ")
    print(f"{'=' * 40}")
    print(f"Матрица A:\n{A}")
    print(f"Начальная точка: {x0}")
    print(f"Шаг α = {alpha:.4f}")

    result = gradient_descent(A, x0, alpha, STOPPING_CRITERIA)

    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    print(f"  Итераций: {result['iterations']}")
    print(f"  x* = {result['x_opt']}")
    print(f"  f(x*) = {result['f_opt']:.6e}")
    print(f"  ||∇f(x*)|| = {result['grad_norm_final']:.6e}")
    print(f"  Сошелся: {result['converged']}")

    # Визуализация для этого одного эксперимента
    if len(x0) == 2:
        single_result = {"Интерактивный эксперимент": result}

        # Один график
        fig1 = plot_contour_with_trajectories(A, single_result)
        fig1.suptitle(f'Градиентный спуск: α={alpha:.4f}, Итераций={result["iterations"]}',
                      fontsize=14, fontweight='bold')

        # Отдельный график
        fig2 = plot_contour_separate_axes(A, single_result)

        plt.show()

    return result


def parameter_sweep(A, x0, alpha_range=(0.01, 1.0), n_points=20):
    """Исследование влияния шага α в заданном диапазоне"""
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
    iterations = []
    converged_flags = []

    print(f"\n📈 СКАНИРОВАНИЕ ПАРАМЕТРА α в диапазоне [{alpha_range[0]}, {alpha_range[1]}]")

    for alpha in tqdm(alphas, desc="Сканирование α"):
        result = gradient_descent(A, x0, alpha, STOPPING_CRITERIA, track_history=False)
        iterations.append(result['iterations'])
        converged_flags.append(result['converged'])

    # Построение графика
    fig, ax = plt.subplots(figsize=(10, 6))

    # Разделение сошедшихся и несошедшихся точек
    alphas_converged = [a for a, c in zip(alphas, converged_flags) if c]
    iters_converged = [i for i, c in zip(iterations, converged_flags) if c]

    alphas_diverged = [a for a, c in zip(alphas, converged_flags) if not c]
    iters_diverged = [i for i, c in zip(iterations, converged_flags) if not c]

    ax.plot(alphas_converged, iters_converged, 'bo-', label='Сошелся', linewidth=2)
    ax.plot(alphas_diverged, iters_diverged, 'rx', label='Не сошелся', markersize=8)

    # Теоретическая граница
    _, alpha_max, _, _ = compute_optimal_alpha(A)
    ax.axvline(x=alpha_max, color='r', linestyle='--',
               label=f'Теоретическая граница α_max = {alpha_max:.3f}')

    ax.set_xlabel('Шаг обучения α', fontsize=12)
    ax.set_ylabel('Число итераций', fontsize=12)
    ax.set_title('Зависимость числа итераций от шага α', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return alphas, iterations, converged_flags


# ==================== ОБНОВЛЕННОЕ МЕНЮ ====================

if __name__ == "__main__":
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА: ГРАДИЕНТНЫЙ СПУСК")
    print("=" * 60)
    print("Выберите режим работы:")
    print("1. Полный набор экспериментов (все графики + анализ)")
    print("2. Только отдельные графики (рекомендуется для анализа)")
    print("3. Интерактивный эксперимент")
    print("4. Сканирование параметра α")
    print("5. Только анализ результатов (без графиков)")

    choice = input("Введите номер (1-5): ").strip()

    if choice == "1":
        # Полный эксперимент с анализом
        results_all, results_points, results_condition = run_complete_experiment()
        plt.show()

    elif choice == "2":
        # Только отдельные графики
        results = run_simple_experiment_with_separate_plots()
        plt.show()

    elif choice == "3":
        # Интерактивный эксперимент
        print("\nДоступные матрицы:")
        for i, (name, matrix) in enumerate(MATRICES.items(), 1):
            print(f"{i}. {name}")

        matrix_choice = int(input("Выберите матрицу (1-3): ")) - 1
        matrix_names = list(MATRICES.keys())
        A_choice = MATRICES[matrix_names[matrix_choice]]

        print("\nДоступные начальные точки:")
        for i, (name, point) in enumerate(INITIAL_POINTS.items(), 1):
            print(f"{i}. {name}: {point}")

        point_choice = int(input("Выберите начальную точку (1-4): ")) - 1
        point_names = list(INITIAL_POINTS.keys())
        x0_choice = INITIAL_POINTS[point_names[point_choice]]

        alpha_input = input("Введите шаг α (или Enter для оптимального): ").strip()
        if alpha_input:
            alpha_choice = float(alpha_input)
        else:
            alpha_choice, _, _, _ = compute_optimal_alpha(A_choice)

        result = interactive_experiment(A_choice, x0_choice, alpha_choice)

    elif choice == "4":
        # Сканирование параметра
        A = MATRICES["Хорошо обусловленная (κ≈2.6)"]
        x0 = INITIAL_POINTS["Далекая от минимума"]

        min_alpha = float(input("Минимальный α (по умолчанию 0.01): ") or "0.01")
        max_alpha = float(input("Максимальный α (по умолчанию 1.0): ") or "1.0")
        n_points = int(input("Количество точек (по умолчанию 20): ") or "20")

        alphas, iterations, converged = parameter_sweep(
            A, x0,
            alpha_range=(min_alpha, max_alpha),
            n_points=n_points
        )

    elif choice == "5":
        # Только анализ
        print("\n📊 ЗАПУСК ТОЛЬКО АНАЛИЗА РЕЗУЛЬТАТОВ")
        print("=" * 50)

        A = MATRICES["Хорошо обусловленная (κ≈2.6)"]
        x0 = INITIAL_POINTS["Далекая от минимума"]

        # Тестируем разные α
        test_alphas = {
            "α = 0.05 (очень маленький)": 0.05,
            "α = 0.1 (маленький)": 0.1,
            "α = 0.4 (оптимальный)": 0.4,
            "α = 0.52 (близкий к пределу)": 0.52,
            "α = 0.6 (сверх предела)": 0.6
        }

        results_alpha = {}
        print("\nЗапуск градиентного спуска для анализа...")
        for label, alpha in test_alphas.items():
            result = gradient_descent(A, x0, alpha, STOPPING_CRITERIA)
            results_alpha[label] = result

        # Тестируем разные матрицы для κ анализа
        results_condition = {}
        x0_fixed = np.array([5.0, 5.0])

        for matrix_name, matrix_A in MATRICES.items():
            if matrix_name != "Случайная симметричная":
                info = analyze_matrix(matrix_A, matrix_name)
                alpha_for_matrix = info['alpha_opt']
                result = gradient_descent(matrix_A, x0_fixed, alpha_for_matrix, STOPPING_CRITERIA)
                results_condition[matrix_name] = result

        # Выводим подробный анализ
        print_detailed_analysis(results_alpha, results_condition, A)

    else:
        print("Неверный выбор. Запускаю полный эксперимент...")
        results_all, results_points, results_condition = run_complete_experiment()
        plt.show()

    print("\n" + "=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА")
    print("=" * 60)