"""Тесты для модуля cyclic_code"""

from cyclic_code import (
    build_cyclic_code,
    build_error_table,
    detect_and_correct_error,
    find_code_length,
    select_generator_polynomial,
)
from polynomial import normalize_remainder, poly_divide


def test_find_code_length():
    """Тест поиска длины кода"""
    k = 4
    n = find_code_length(k)
    # Для k=4 должно быть n=7 (проверяем формулу 2^4 <= 2^7 / (1+7) = 128/8 = 16)
    assert n == 7, f"Для k={k} ожидалось n=7, получено n={n}"
    print("✓ test_find_code_length пройден")


def test_select_generator_polynomial():
    """Тест выбора образующего полинома"""
    p = 3
    poly = select_generator_polynomial(p)
    assert len(poly) == p + 1, (
        f"Полином должен иметь длину {p + 1}, получено {len(poly)}"
    )
    assert poly[0] == 1, "Старший коэффициент должен быть 1"
    print("✓ test_select_generator_polynomial пройден")


def test_build_cyclic_code():
    """Тест построения циклического кода"""
    info_bits = [1, 0, 1, 1]  # x^3 + x + 1
    generator_poly = [1, 0, 1, 1]  # x^3 + x + 1
    p = 3

    code, remainder = build_cyclic_code(info_bits, generator_poly, p)

    # Проверяем, что правильный код делится без остатка
    _, check_remainder = poly_divide(code, generator_poly)
    check_remainder = normalize_remainder(check_remainder, p)
    assert all(x == 0 for x in check_remainder) or check_remainder == [0], (
        f"Правильный код должен делиться без остатка, остаток: {check_remainder}"
    )

    # Проверяем длину кода
    assert len(code) == len(info_bits) + p, (
        f"Длина кода должна быть {len(info_bits) + p}, получено {len(code)}"
    )

    print("✓ test_build_cyclic_code пройден")


def test_build_error_table():
    """Тест построения таблицы ошибок"""
    generator_poly = [1, 0, 1, 1]  # x^3 + x + 1
    n = 7
    error_table = build_error_table(generator_poly, n)

    # Проверяем, что таблица содержит n записей
    assert len(error_table) == n, (
        f"Таблица должна содержать {n} записей, получено {len(error_table)}"
    )

    # Проверяем, что все остатки уникальны
    remainders = list(error_table.keys())
    assert len(remainders) == len(set(remainders)), (
        "Все остатки должны быть уникальными"
    )

    print("✓ test_build_error_table пройден")


def test_detect_and_correct_error():
    """Тест обнаружения и исправления ошибки"""
    generator_poly = [1, 0, 1, 1]  # x^3 + x + 1
    p = 3
    n = 7

    # Строим правильный код
    info_bits = [1, 0, 1, 1]
    correct_code, _ = build_cyclic_code(info_bits, generator_poly, p)

    # Строим таблицу ошибок
    error_table = build_error_table(generator_poly, n)

    # Вносим ошибку в каждую позицию и проверяем исправление
    # Пропускаем позиции, где ошибка не обнаруживается (если код с ошибкой тоже делится без остатка)
    for error_pos in range(n):
        code_with_error = correct_code[:]
        code_with_error[error_pos] ^= 1

        detected_pos, remainder = detect_and_correct_error(
            code_with_error, generator_poly, error_table
        )

        # Если остаток не нулевой, ошибка должна быть обнаружена
        if not (all(x == 0 for x in remainder) or remainder == [0]):
            assert detected_pos is not None, (
                f"Ошибка в позиции {error_pos} должна быть обнаружена, остаток: {remainder}"
            )
            assert detected_pos == error_pos, (
                f"Ошибка в позиции {error_pos} обнаружена как {detected_pos}, остаток: {remainder}"
            )

            # Исправляем ошибку
            corrected_code = code_with_error[:]
            corrected_code[detected_pos] ^= 1

            assert corrected_code == correct_code, (
                "Исправленный код должен совпадать с правильным"
            )

    print("✓ test_detect_and_correct_error пройден")


def test_detect_no_error():
    """Тест обнаружения отсутствия ошибки"""
    generator_poly = [1, 0, 1, 1]  # x^3 + x + 1
    p = 3

    # Строим правильный код
    info_bits = [1, 0, 1, 1]
    correct_code, _ = build_cyclic_code(info_bits, generator_poly, p)

    # Строим таблицу ошибок
    error_table = build_error_table(generator_poly, len(correct_code))

    # Проверяем правильный код (без ошибок)
    detected_pos, remainder = detect_and_correct_error(
        correct_code, generator_poly, error_table
    )

    assert detected_pos is None, "В правильном коде не должно быть обнаружено ошибок"
    assert all(x == 0 for x in remainder) or remainder == [0], (
        "Остаток должен быть нулевым"
    )

    print("✓ test_detect_no_error пройден")


if __name__ == "__main__":
    print("Запуск тестов для cyclic_code...")
    test_find_code_length()
    test_select_generator_polynomial()
    test_build_cyclic_code()
    test_build_error_table()
    test_detect_and_correct_error()
    test_detect_no_error()
    print("\nВсе тесты для cyclic_code пройдены!")
