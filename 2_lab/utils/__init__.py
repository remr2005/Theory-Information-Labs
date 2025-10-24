"""
Модуль различных утилит
"""

from .calculate_entropy_module import (
    calculate_conditional_entropy,
    calculate_entropy,
    calculate_mutual_information,
    calculate_private_entropy,
)
from .makearray import (
    calc_joint_prob,
    calc_p_y,
    make_array_input,
    make_matrix_transition,
)
