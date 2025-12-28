"""Запуск всех тестов"""

import sys

def run_all_tests():
    """Запускает все тесты"""
    print("=" * 70)
    print("ЗАПУСК ВСЕХ ТЕСТОВ")
    print("=" * 70)
    print()
    
    # Тесты для polynomial
    print("1. Тесты для модуля polynomial:")
    print("-" * 70)
    try:
        import test_polynomial
        test_polynomial.test_poly_add()
        test_polynomial.test_poly_multiply()
        test_polynomial.test_poly_divide()
        test_polynomial.test_poly_divide_zero()
        test_polynomial.test_normalize_remainder()
        test_polynomial.test_poly_to_string()
        print("✓ Все тесты для polynomial пройдены!")
    except Exception as e:
        print(f"✗ Ошибка в тестах polynomial: {e}")
        return False
    
    print()
    
    # Тесты для cyclic_code
    print("2. Тесты для модуля cyclic_code:")
    print("-" * 70)
    try:
        import test_cyclic_code
        test_cyclic_code.test_find_code_length()
        test_cyclic_code.test_select_generator_polynomial()
        test_cyclic_code.test_build_cyclic_code()
        test_cyclic_code.test_build_error_table()
        test_cyclic_code.test_detect_and_correct_error()
        test_cyclic_code.test_detect_no_error()
        print("✓ Все тесты для cyclic_code пройдены!")
    except Exception as e:
        print(f"✗ Ошибка в тестах cyclic_code: {e}")
        return False
    
    print()
    print("=" * 70)
    print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)




