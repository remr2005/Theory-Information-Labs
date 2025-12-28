"""Тесты для модуля polynomial"""

from polynomial import poly_add, poly_multiply, poly_divide, normalize_remainder, poly_to_string


def test_poly_add():
    """Тест сложения полиномов"""
    # x^2 + x + 1 + x + 1 = x^2
    poly1 = [1, 1, 1]  # x^2 + x + 1
    poly2 = [0, 1, 1]  # x + 1
    result = poly_add(poly1, poly2)
    expected = [1, 0, 0]  # x^2
    assert result == expected, f"Ожидалось {expected}, получено {result}"
    print("✓ test_poly_add пройден")


def test_poly_multiply():
    """Тест умножения полиномов"""
    # (x + 1) * (x + 1) = x^2 + 1
    poly1 = [1, 1]  # x + 1
    poly2 = [1, 1]  # x + 1
    result = poly_multiply(poly1, poly2)
    expected = [1, 0, 1]  # x^2 + 1
    assert result == expected, f"Ожидалось {expected}, получено {result}"
    print("✓ test_poly_multiply пройден")


def test_poly_divide():
    """Тест деления полиномов"""
    # x^3 / (x^2 + x + 1) = x, остаток x^2 + x
    dividend = [1, 0, 0, 0]  # x^3
    divisor = [1, 1, 1]  # x^2 + x + 1
    quotient, remainder = poly_divide(dividend, divisor)
    expected_quotient = [1]  # x
    expected_remainder = [1, 1, 0]  # x^2 + x
    assert quotient == expected_quotient, f"Частное: ожидалось {expected_quotient}, получено {quotient}"
    assert remainder == expected_remainder, f"Остаток: ожидалось {expected_remainder}, получено {remainder}"
    print("✓ test_poly_divide пройден")


def test_poly_divide_zero():
    """Тест деления на ноль"""
    dividend = [1, 0, 1]
    divisor = [0]
    try:
        poly_divide(dividend, divisor)
        assert False, "Должно было быть исключение"
    except ValueError:
        print("✓ test_poly_divide_zero пройден")


def test_normalize_remainder():
    """Тест нормализации остатка"""
    remainder = [1, 1, 0]  # x^2 + x
    p = 7
    result = normalize_remainder(remainder, p)
    expected = [0, 0, 0, 0, 0, 1, 1, 0][-p:]  # Должно быть [0, 0, 0, 0, 1, 1, 0]
    assert len(result) == p, f"Длина должна быть {p}, получено {len(result)}"
    assert result == [0, 0, 0, 0, 1, 1, 0], f"Ожидалось [0, 0, 0, 0, 1, 1, 0], получено {result}"
    print("✓ test_normalize_remainder пройден")


def test_poly_to_string():
    """Тест преобразования полинома в строку"""
    poly = [1, 0, 1, 1]  # x^3 + x + 1
    result = poly_to_string(poly)
    assert "x^3" in result
    assert "x" in result
    assert "1" in result
    print("✓ test_poly_to_string пройден")


if __name__ == "__main__":
    print("Запуск тестов для polynomial...")
    test_poly_add()
    test_poly_multiply()
    test_poly_divide()
    test_poly_divide_zero()
    test_normalize_remainder()
    test_poly_to_string()
    print("\nВсе тесты для polynomial пройдены!")




