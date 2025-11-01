"""
Модуль для вычисления энтропии и связанных характеристик
"""

import numpy as np
from numpy import ndarray


def calculate_entropy(probabilities: ndarray) -> float:
    """
    Вычисление энтропии H(X) для массива вероятностей.
    H(X) = -Σ p(xi) * log₂ p(xi)

    Args:
        probabilities (ndarray): Массив вероятностей событий (P(X)).

    Returns:
        float: Энтропия в битах.
    """
    probabilities = np.array(probabilities)
    # Игнорируем нулевые вероятности, чтобы не было log(0)
    mask = probabilities > 0
    return -np.sum(probabilities[mask] * np.log2(probabilities[mask]))


def calculate_conditional_probability_X_given_Y(P_joint: ndarray, P_Y: ndarray) -> ndarray:
    """
    Вычисление матрицы условных вероятностей P(X|Y) из совместных вероятностей.
    P(X|Y)[i,j] = P(X=i, Y=j) / P(Y=j)
    
    Args:
        P_joint (ndarray): Матрица совместных вероятностей P(X,Y) размером NxN
                          где P_joint[i,j] = P(X=i, Y=j)
        P_Y (ndarray): Массив вероятностей P(Y) на выходе
        
    Returns:
        ndarray: Матрица условных вероятностей P(X|Y) размером NxN
                 где P_X_given_Y[i,j] = P(X=i | Y=j)
    """
    # Избегаем деления на ноль
    P_Y_safe = np.where(P_Y > 0, P_Y, 1.0)
    # P(X|Y)[i,j] = P_joint[i,j] / P_Y[j]
    # Используем broadcasting: делим каждую колонку на соответствующий элемент P_Y
    P_X_given_Y = P_joint / P_Y_safe[np.newaxis, :]
    return P_X_given_Y


def calculate_conditional_entropy(P_joint: ndarray, P_X_given_Y: ndarray) -> float:
    """
    Вычисление условной энтропии H(X|Y).
    H(X|Y) = -ΣΣ p(xi,yj) log₂ p(xi|yj)

    Args:
        P_joint (ndarray): Матрица совместных вероятностей P(X,Y) размером NxN.
        P_X_given_Y (ndarray): Матрица условных вероятностей P(X|Y) размером NxN.

    Returns:
        float: Условная энтропия в битах.
    """
    mask = P_X_given_Y > 0
    return -np.sum(P_joint[mask] * np.log2(P_X_given_Y[mask]))


def calculate_average_duration(Px: ndarray, Tx: ndarray) -> float:
    """
    Расчет средней длительности символа.
    тау = Σ Px[i] * Tx[i]
    
    Args:
        Px (ndarray): Массив вероятностей символов
        Tx (ndarray): Массив длительностей символов (мкс)
        
    Returns:
        float: Средняя длительность в микросекундах
    """
    return np.sum(Px * Tx)


def calculate_joint_probability(Px: ndarray, P_xy: ndarray) -> ndarray:
    """
    Расчет матрицы совместных вероятностей P(X,Y).
    P(xi, yj) = P(xi) * P(yj|xi)
    
    Args:
        Px (ndarray): Массив вероятностей на входе
        P_xy (ndarray): Матрица переходных вероятностей
        
    Returns:
        ndarray: Матрица совместных вероятностей
    """
    return P_xy * Px[:, np.newaxis]


def calculate_output_probabilities(Px: ndarray, P_xy: ndarray) -> ndarray:
    """
    Расчет вероятностей на выходе P(Y).
    p(yj) = Σ P(xi) * P(yj|xi)
    
    Args:
        Px (ndarray): Массив вероятностей на входе
        P_xy (ndarray): Матрица переходных вероятностей
        
    Returns:
        ndarray: Массив вероятностей на выходе
    """
    return Px @ P_xy

