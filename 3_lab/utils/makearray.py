"""
Функции для генерации массивов и матриц для лабораторной работы №3
"""

import numpy as np
from numpy import ndarray


def make_array_probabilities(n: int) -> ndarray:
    """
    Создает массив вероятностей появления сообщений на входе Px[i]
    
    Args:
        n (int): Количество символов
        
    Returns:
        ndarray: Массив вероятностей длины N
    """
    return np.random.dirichlet(np.ones(n))


def make_array_durations(n: int) -> ndarray:
    """
    Генерирует массив длительностей символов Tx[i] в диапазоне (0, N] мкс
    
    Args:
        n (int): Количество символов
        
    Returns:
        ndarray: Массив длительностей в микросекундах
    """
    return np.random.uniform(0, n, n)


def make_matrix_error_probabilities(n: int) -> ndarray:
    """
    Генерирует матрицу вероятностей ошибок P[X, Y] размером NxN.
    Диагональные элементы (P[i][i]) должны быть >= 0.70.
    Остальные элементы строки распределяются случайно так, чтобы сумма строки = 1.
    
    Args:
        n (int): Количество символов
        
    Returns:
        ndarray: Матрица переходных вероятностей размером (n x n)
                где сумма каждой строки равна 1 и диагональные элементы >= 0.70
    """
    q = 1 / (2 * n)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        # Генерируем диагональный элемент от 0.70 до 1.0
        diagonal_element = np.random.uniform(0.70, 1.0)
        
        # Оставшаяся вероятность для остальных элементов
        remaining_prob = 1.0 - diagonal_element
        
        # Генерируем случайные значения для всех элементов строки, кроме диагонального
        row_values = np.random.uniform(0, q, n)
        row_values[i] = 0  # Обнуляем диагональный элемент
        
        # Нормализуем остальные элементы так, чтобы их сумма = remaining_prob
        non_diagonal_sum = np.sum(row_values)
        if non_diagonal_sum > 0:
            row_values = row_values / non_diagonal_sum * remaining_prob
        else:
            # Если все нули, распределяем равномерно
            row_values = np.ones(n) * (remaining_prob / (n - 1))
            row_values[i] = 0
        
        # Устанавливаем диагональный элемент
        row_values[i] = diagonal_element
        
        # Сохраняем строку в матрицу
        matrix[i, :] = row_values
    
    return matrix

