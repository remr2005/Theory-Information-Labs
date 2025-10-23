"""
Функция для создния случайнного массива длинной N
"""

import numpy as np
from numpy import ndarray


def make_array_input(n: int) -> ndarray:
    """
    Создает массив вероятностей появления того или иного сообщения
    """

    return np.random.dirichlet(np.ones(n))


def make_matrix_transition(n: int) -> ndarray:
    """
    Генерация матрицы переходных вероятностей P(X/Y) для n дискретных сообщений.

    Диагональные элементы задают вероятность безошибочной передачи
    (случайное значение от 0.7 до 1). Остальные элементы строки —
    вероятности ошибочной передачи, нормированные так, чтобы сумма
    каждой строки была равна 1.

    Args:
        n (int): Количество дискретных сообщений на входе и выходе.

    Returns:
        np.ndarray: Матрица переходных вероятностей размером (n x n),
                    каждая строка которой суммируется в 1.
    """
    diag = 0.7 + 0.3 * np.random.rand(n)

    remaining = 1 - diag

    errors = np.random.rand(n, n)
    np.fill_diagonal(errors, 0)

    row_sums = errors.sum(axis=1, keepdims=True)
    errors = errors / row_sums * remaining[:, np.newaxis]

    return errors + np.diag(diag)


def calc_p_y(p_x: ndarray, P_xy: ndarray) -> ndarray:
    """
    Рассчет вероятностей появления сообщений на выходе P(Y)
    """
    return p_x @ P_xy  # матричное умножение


def calc_joint_prob(p_x: ndarray, P_xy: ndarray) -> ndarray:
    """
    Рассчет матрицы совместных вероятностей P(X,Y)
    """
    p_y = calc_p_y(p_x, P_xy)
    return P_xy * p_y  # broadcasting
