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
    Элементы (кроме первого в каждой строке) в диапазоне (0, q], где q = 1/(2N).
    P(x_i, y_1) = 1 - sum from j=2 to N p(x_i, y_j) для каждой строки i
    
    Args:
        n (int): Количество символов
        
    Returns:
        ndarray: Матрица переходных вероятностей размером (n x n)
                где сумма каждой строки равна 1
    """
    q = 1 / (2 * n)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        # Генерируем случайные значения для элементов строки со 2 по N
        # Ограничиваем их сумму так, чтобы первый элемент был положительным
        max_sum = min(0.99, q * (n - 1) * 0.8)  # Оставляем место для первого элемента
        
        # Генерируем значения в (0, q]
        row_values = np.random.uniform(0, q, n - 1)
        
        # Нормализуем, чтобы сумма не превышала max_sum
        current_sum = np.sum(row_values)
        if current_sum > max_sum:
            row_values = row_values / current_sum * max_sum
        
        # Первый элемент строки: 1 - сумма остальных
        first_element = 1 - np.sum(row_values)
        
        matrix[i, 0] = first_element
        matrix[i, 1:] = row_values
    
    return matrix

