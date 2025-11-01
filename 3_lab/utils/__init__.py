"""
Модуль утилит для лабораторной работы №3
"""

from utils.calculate_entropy_module import (
    calculate_average_duration,
    calculate_conditional_entropy,
    calculate_conditional_probability_X_given_Y,
    calculate_entropy,
    calculate_joint_probability,
    calculate_output_probabilities,
)
from utils.makearray import (
    make_array_durations,
    make_array_probabilities,
    make_matrix_error_probabilities,
)

__all__ = [
    "make_array_probabilities",
    "make_array_durations",
    "make_matrix_error_probabilities",
    "calculate_entropy",
    "calculate_conditional_entropy",
    "calculate_conditional_probability_X_given_Y",
    "calculate_average_duration",
    "calculate_joint_probability",
    "calculate_output_probabilities",
]

