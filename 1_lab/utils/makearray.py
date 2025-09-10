"""
Функция для создния случайнного массива длинной N
"""

import numpy as np


def make_array(n: int) -> list[float]:
    """
    Создает массив вероятностей появления того или иного сообщения
    """

    return list(np.random.dirichlet(np.ones(n)))
