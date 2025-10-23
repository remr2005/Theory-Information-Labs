import numpy as np
from numpy import ndarray


def calculate_entropy(probabilities: ndarray) -> float:
    """
    Вычисление энтропии H(X) для массива вероятностей.

    Args:
        probabilities (ndarray): Массив вероятностей событий (P(X)).

    Returns:
        float: Энтропия в битах.
    """
    probabilities = np.array(probabilities)
    # игнорируем нулевые вероятности, чтобы не было log(0)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))


def calculate_private_entropy(probability: float) -> float:
    """
    Вычисление частной энтропии H(x_i) для одного события.

    Args:
        probability (float): Вероятность события.

    Returns:
        float: Частная энтропия в битах.
    """
    if probability <= 0:
        return 0.0
    return np.log2(1 / probability)


def calculate_conditional_entropy(P_xy: ndarray, P_joint: ndarray) -> float:
    """
    Вычисление условной энтропии H(X|Y) при неполной достоверности передачи.

    Args:
        P_xy (ndarray): Матрица условных вероятностей P(X|Y) размером NxN.
        P_joint (ndarray): Матрица совместных вероятностей P(X,Y) размером NxN.

    Returns:
        float: Условная энтропия в битах.
    """
    mask = P_xy > 0
    return -np.sum(P_joint[mask] * np.log2(P_xy[mask]))


def calculate_mutual_information(H_x: float, H_x_given_y: float) -> float:
    """
    Вычисление среднего количества информации I(X,Y) при неполной достоверности.

    Args:
        H_x (float): Энтропия на входе H(X).
        H_x_given_y (float): Условная энтропия H(X|Y).

    Returns:
        float: Среднее количество информации I(X,Y) в битах.
    """
    return H_x - H_x_given_y
