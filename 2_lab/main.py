"""Main file - Лабораторная работа №2: Передача информации при наличии помех"""

import numpy as np
from utils import (
    calc_joint_prob,
    calc_p_y,
    calculate_conditional_entropy,
    calculate_entropy,
    calculate_mutual_information,
    make_array_input,
    make_matrix_transition,
)


def format_array(arr, max_elements=9):
    """Форматирует массив с обрезанием для больших размеров"""
    if len(arr) <= max_elements:
        return [round(float(p), 4) for p in arr]
    else:
        start = [round(float(p), 4) for p in arr[:4]]
        end = [round(float(p), 4) for p in arr[-4:]]
        return start + ["..."] + end


def format_matrix(matrix, max_elements=9, max_rows=7):
    """Форматирует матрицу с обрезанием для больших размеров"""
    # Обрезаем строки если их больше max_rows
    if matrix.shape[0] > max_rows:
        rows_to_show = matrix[:3]  # первые 3 строки
        rows_to_show = np.vstack([rows_to_show, matrix[-3:]])  # последние 3 строки
    else:
        rows_to_show = matrix

    result = []
    for i, row in enumerate(rows_to_show):
        if matrix.shape[0] > max_rows and i == 3:  # добавляем "..." между строками
            result.append(
                ["..."] * (9 if matrix.shape[1] > max_elements else matrix.shape[1])
            )
        elif matrix.shape[1] <= max_elements:
            result.append([round(float(p), 4) for p in row])
        else:
            start = [round(float(p), 4) for p in row[:4]]
            end = [round(float(p), 4) for p in row[-4:]]
            result.append(start + ["..."] + end)

    return result


# Глобальные переменные для хранения результатов экспериментов
experiment_results = []


def conduct_experiment(experiment_num: int, n: int = 61) -> dict:
    """
    Проведение одного эксперимента с передачей информации через канал с помехами

    Args:
        experiment_num (int): Номер эксперимента
        n (int): Количество дискретных сообщений

    Returns:
        dict: Результаты эксперимента
    """
    print(f"\n{'=' * 60}")
    print(f"ЭКСПЕРИМЕНТ {experiment_num}")
    print(f"{'=' * 60}")

    # а) Генерация массива вероятностей P(X) на входе
    print("\nа) Генерация массива вероятностей P(X) на входе:")
    P_X = make_array_input(n)
    formatted_P_X = format_array(P_X)
    print(f"P(X) = {formatted_P_X}")
    if len(P_X) > 9:
        print("(массив обрезан для удобства отображения)")

    # б) Генерация матрицы переходных вероятностей P(X|Y)
    print("\nб) Генерация матрицы переходных вероятностей P(X|Y):")
    P_XY = make_matrix_transition(n)
    print("Матрица P(X|Y):")
    formatted_P_XY = format_matrix(P_XY)
    for row in formatted_P_XY:
        print(f"{row}")
    if n > 7:
        print("(матрица обрезана для удобства отображения)")

    # в) Расчет вероятностей P(Y) на выходе
    print("\nв) Расчет вероятностей P(Y) на выходе:")
    P_Y = calc_p_y(P_X, P_XY)
    formatted_P_Y = format_array(P_Y)
    print(f"P(Y) = {formatted_P_Y}")
    if len(P_Y) > 9:
        print("(массив обрезан для удобства отображения)")

    # г) Расчет матрицы совместных вероятностей P(X,Y)
    print("\nг) Расчет матрицы совместных вероятностей P(X,Y):")
    P_joint = calc_joint_prob(P_X, P_XY)
    print("Матрица P(X,Y):")
    formatted_P_joint = format_matrix(P_joint)
    for row in formatted_P_joint:
        print(f"{row}")
    if n > 7:
        print("(матрица обрезана для удобства отображения)")

    # д) Энтропия на входе H(X)
    print("\nд) Энтропия на входе H(X):")
    H_X = calculate_entropy(P_X)
    print(f"H(X) = {round(H_X, 4)} бит")

    # е) Условная энтропия H(X|Y)
    print("\nе) Условная энтропия H(X|Y):")
    H_X_given_Y = calculate_conditional_entropy(P_XY, P_joint)
    print(f"H(X|Y) = {round(H_X_given_Y, 4)} бит")

    # ж) Количество информации при неполной достоверности I(X,Y)
    print("\nж) Количество информации при неполной достоверности I(X,Y):")
    I_XY = calculate_mutual_information(H_X, H_X_given_Y)
    print(f"I(X,Y) = {round(I_XY, 4)} бит")

    # Сохранение результатов
    result = {
        "experiment_num": experiment_num,
        "P_X": P_X,
        "P_XY": P_XY,
        "P_Y": P_Y,
        "P_joint": P_joint,
        "H_X": H_X,
        "H_X_given_Y": H_X_given_Y,
        "I_XY": I_XY,
        "efficiency": I_XY / H_X * 100,
    }

    return result


def main() -> None:
    """
    Основная функция для проведения 6 экспериментов
    """
    print("ЛАБОРАТОРНАЯ РАБОТА №2")
    print("Передача информации при наличии помех")
    print("=" * 60)

    # Проведение 6 экспериментов
    for i in range(6):
        result = conduct_experiment(i + 1)
        experiment_results.append(result)

    # Задание II: Расчет среднего количества информации
    print(f"\n{'=' * 60}")
    print("ЗАДАНИЕ II: СРЕДНЕЕ КОЛИЧЕСТВО ИНФОРМАЦИИ")
    print(f"{'=' * 60}")

    # Извлечение результатов
    I_XY_values = [r["I_XY"] for r in experiment_results]

    # Расчет среднего значения
    avg_I_XY = np.mean(I_XY_values)

    print(f"\nСреднее количество информации I(X,Y): {round(avg_I_XY, 4)} бит")

    print(f"\n{'=' * 60}")
    print("ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
