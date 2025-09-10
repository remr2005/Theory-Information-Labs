"""
Вычисление энтропии
"""

from math import log2


def calculate_entropy(probability_array: list[float]) -> float:
    """
    Функция для вычисления энтропии
    """
    return -sum([p * log2(p) for p in probability_array])


def calculate_private_entropy(probability: float) -> float:
    """
    Функция для вычилсения частной энтропии
    """
    return log2(1 / probability)
