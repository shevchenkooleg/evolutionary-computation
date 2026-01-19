Описание проекта

Проект содержит лабораторные работы по исследованию эволюционных алгоритмов на Python с отчетами в LaTeX:

Градиентный спуск - минимизация квадратичной формы  
Генетический алгоритм - минимизация функции Растригина  
Алгоритм оптимизации роем частиц (PSO) - минимизация функции Растригина

Системные требования:

Python 3.10+ (рекомендуется Python 3.14)  
TeX Live 2023+ или MiKTeX (для компиляции отчетов)

Установка зависимостей

```
pip install numpy matplotlib seaborn tqdm
```


Структура проекта

```
evolutionary-algorithms/
├── lab_1/  # Градиентный спуск
│   ├── gradient_descent.py
│   └── lab_1_report.tex
├── lab_2/  # Генетический алгоритм
│   ├── genetic_algorithm.py
│   └── lab_2_report.tex
├── lab_3/  # Алгоритм PSO
│   ├── pso_algorithm.py
│   └── lab_3_report.tex
├── README.md
├── .gitignore
└── LICENSE
```
