"""Модуль для работы с циклическими кодами"""

from polynomial import poly_divide, normalize_remainder

# Таблица неприводимых полиномов по степеням
IRREDUCIBLE_POLYNOMIALS = {
    1: [1, 1],  # x + 1
    2: [1, 1, 1],  # x^2 + x + 1
    3: [[1, 0, 1, 1], [1, 1, 0, 1]],  # x^3 + x + 1, x^3 + x^2 + 1
    4: [[1, 0, 0, 1, 1], [1, 0, 1, 0, 1]],  # x^4 + x + 1, x^4 + x^2 + 1
    5: [
        [1, 0, 1, 0, 0, 1],  # x^5 + x^2 + 1
        [1, 0, 0, 1, 0, 1],  # x^5 + x^3 + 1
        [1, 0, 1, 1, 1, 1],  # x^5 + x^3 + x^2 + x + 1
        [1, 1, 0, 1, 1, 1],  # x^5 + x^4 + x^2 + x + 1
    ],
    6: [
        [1, 0, 0, 0, 0, 1, 1],  # x^6 + x + 1
        [1, 0, 0, 1, 0, 0, 1],  # x^6 + x^3 + 1
        [1, 0, 0, 0, 1, 1, 1],  # x^6 + x^5 + 1
        [1, 0, 1, 0, 1, 1, 1],  # x^6 + x^4 + x^2 + x + 1
    ],
    7: [
        [1, 0, 0, 0, 0, 0, 1, 1],  # x^7 + x + 1
        [1, 0, 0, 1, 0, 0, 0, 1],  # x^7 + x^3 + 1
        [1, 0, 0, 0, 1, 0, 0, 1],  # x^7 + x^4 + 1
        [1, 0, 0, 1, 1, 1, 1, 1],  # x^7 + x^3 + x^2 + x + 1
    ],
    8: [
        [1, 0, 0, 0, 1, 1, 0, 1, 1],  # x^8 + x^4 + x^3 + x + 1
        [1, 0, 0, 0, 1, 1, 1, 1, 1],  # x^8 + x^4 + x^3 + x^2 + 1
        [1, 0, 0, 1, 0, 1, 0, 1, 1],  # x^8 + x^5 + x^3 + x + 1
        [1, 0, 0, 1, 0, 1, 1, 1, 1],  # x^8 + x^5 + x^3 + x^2 + 1
    ],
}


def find_code_length(k):
    """Находит минимальную длину кода n из формулы 2^k <= 2^n / (1+n)"""
    n = k
    while True:
        if 2**k <= (2**n) / (1 + n):
            return n
        n += 1


def select_generator_polynomial(p):
    """Выбирает образующий полином степени p"""
    if p in IRREDUCIBLE_POLYNOMIALS:
        polys = IRREDUCIBLE_POLYNOMIALS[p]
        if isinstance(polys[0], list):
            return polys[0]  # Берем первый из списка
        else:
            return polys
    raise ValueError(f"Нет неприводимого полинома степени {p}")


def build_cyclic_code(info_bits, generator_poly, p):
    """Строит циклический код из информационных битов"""
    # G(x) - полином информационных битов
    G = info_bits[:]

    # Умножаем G(x) на x^p (добавляем p нулей справа)
    G_shifted = G + [0] * p

    # Делим x^p * G(x) на P(x) и получаем остаток R(x)
    _, remainder = poly_divide(G_shifted, generator_poly)

    # Нормализуем остаток
    remainder = normalize_remainder(remainder, p)

    # Циклический код: информационные биты + остаток
    cyclic_code = info_bits + remainder

    return cyclic_code, remainder


def build_error_table(generator_poly, n):
    """Строит таблицу соответствия позиции ошибки и остатка"""
    error_table = {}
    p = len(generator_poly) - 1
    for error_pos in range(n):
        # Создаем полином с ошибкой в позиции error_pos
        # Позиция error_pos в коде соответствует x^(n-1-error_pos)
        # Полином ошибки: единица в позиции error_pos
        error_poly = [0] * error_pos + [1] + [0] * (n - error_pos - 1)
        # Делим на образующий полином
        _, remainder = poly_divide(error_poly, generator_poly)
        # Нормализуем остаток
        remainder = normalize_remainder(remainder, p)
        # Сохраняем в таблице (ключ - строка остатка, значение - позиция ошибки)
        remainder_str = "".join(map(str, remainder))
        if remainder_str in error_table:
            print(f"ВНИМАНИЕ: Дублирование остатка для позиций {error_table[remainder_str]} и {error_pos}")
        error_table[remainder_str] = error_pos
    return error_table


def detect_and_correct_error(received_code, generator_poly, error_table):
    """Обнаруживает и исправляет ошибку в принятом коде"""
    # Делим принятый код на образующий полином
    _, remainder = poly_divide(received_code, generator_poly)

    # Нормализуем остаток
    p = len(generator_poly) - 1
    remainder = normalize_remainder(remainder, p)

    remainder_str = "".join(map(str, remainder))

    # Проверяем, есть ли ошибка (остаток не нулевой)
    if all(x == 0 for x in remainder):
        return None, remainder  # Ошибок нет

    # Ищем позицию ошибки в таблице
    if remainder_str in error_table:
        error_pos = error_table[remainder_str]
        # Исправляем ошибку
        corrected_code = received_code[:]
        corrected_code[error_pos] ^= 1
        return error_pos, remainder

    return None, remainder  # Ошибка не найдена в таблице

