"""Модуль для работы с полиномами по модулю 2"""


def poly_to_string(poly):
    """Преобразует полином в строку для вывода"""
    if not poly or all(x == 0 for x in poly):
        return "0"
    terms = []
    for i, coeff in enumerate(poly):
        if coeff == 1:
            power = len(poly) - 1 - i
            if power == 0:
                terms.append("1")
            elif power == 1:
                terms.append("x")
            else:
                terms.append(f"x^{power}")
    return " + ".join(terms) if terms else "0"


def poly_add(poly1, poly2):
    """Сложение полиномов по модулю 2"""
    max_len = max(len(poly1), len(poly2))
    result = [0] * max_len
    for i in range(len(poly1)):
        result[i + max_len - len(poly1)] ^= poly1[i]
    for i in range(len(poly2)):
        result[i + max_len - len(poly2)] ^= poly2[i]
    # Убираем ведущие нули
    while result and result[0] == 0:
        result.pop(0)
    return result if result else [0]


def poly_multiply(poly1, poly2):
    """Умножение полиномов по модулю 2"""
    if not poly1 or not poly2:
        return [0]
    result = [0] * (len(poly1) + len(poly2) - 1)
    for i, coeff1 in enumerate(poly1):
        if coeff1 == 1:
            for j, coeff2 in enumerate(poly2):
                if coeff2 == 1:
                    result[i + j] ^= 1
    # Убираем ведущие нули
    while result and result[0] == 0:
        result.pop(0)
    return result if result else [0]


def poly_divide(dividend, divisor):
    """Деление полиномов по модулю 2, возвращает (частное, остаток)"""
    if not divisor or all(x == 0 for x in divisor):
        raise ValueError("Деление на ноль")
    if not dividend or all(x == 0 for x in dividend):
        return ([0], [0])

    # Убираем ведущие нули
    while dividend and dividend[0] == 0:
        dividend = dividend[1:]
    while divisor and divisor[0] == 0:
        divisor = divisor[1:]
    
    if not dividend or all(x == 0 for x in dividend):
        return ([0], [0])
    if not divisor or all(x == 0 for x in divisor):
        raise ValueError("Деление на ноль")

    dividend = dividend[:]
    quotient = []
    divisor_len = len(divisor)

    # Деление: пока степень делимого >= степени делителя
    while len(dividend) >= divisor_len:
        if dividend[0] == 1:
            quotient.append(1)
            # Вычитаем (XOR) делитель
            for i in range(divisor_len):
                if i < len(dividend):
                    dividend[i] ^= divisor[i]
        else:
            quotient.append(0)
            # Если старший бит 0, просто удаляем его
            if dividend:
                dividend.pop(0)
            continue
        
        # Убираем ведущие нули
        while dividend and dividend[0] == 0:
            dividend.pop(0)
        
        # Останавливаемся, если степень стала строго меньше
        if not dividend or len(dividend) < divisor_len:
            break
        
        # Если длина равна, но остаток меньше делителя, останавливаемся
        if len(dividend) == divisor_len:
            # Сравниваем остаток с делителем
            if dividend < divisor:
                break

    # Убираем ведущие нули из остатка
    while dividend and dividend[0] == 0:
        dividend.pop(0)

    remainder = dividend if dividend else [0]
    # Убираем ведущие нули из частного
    while quotient and quotient[0] == 0:
        quotient.pop(0)

    return (quotient if quotient else [0], remainder)


def normalize_remainder(remainder, p):
    """Нормализует остаток до длины p (добавляет ведущие нули)"""
    remainder = remainder[:]
    # Убираем ведущие нули
    while remainder and remainder[0] == 0:
        remainder.pop(0)
    if not remainder:
        remainder = [0]
    # Дополняем до длины p ведущими нулями
    while len(remainder) < p:
        remainder = [0] + remainder
    # Обрезаем до длины p (берем младшие p коэффициентов)
    remainder = remainder[-p:] if len(remainder) > p else remainder
    return remainder

