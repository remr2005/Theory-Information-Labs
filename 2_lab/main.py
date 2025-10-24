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
    print(f"P(X) = {[round(float(p), 3) for p in P_X]}")

    # б) Генерация матрицы переходных вероятностей P(X|Y)
    print("\nб) Генерация матрицы переходных вероятностей P(X|Y):")
    P_XY = make_matrix_transition(n)
    print("Матрица P(X|Y):")
    for row in P_XY:
        print(f"{[round(float(p), 3) for p in row]}")

    # в) Расчет вероятностей P(Y) на выходе
    print("\nв) Расчет вероятностей P(Y) на выходе:")
    P_Y = calc_p_y(P_X, P_XY)
    print(f"P(Y) = {[round(float(p), 3) for p in P_Y]}")

    # г) Расчет матрицы совместных вероятностей P(X,Y)
    print("\nг) Расчет матрицы совместных вероятностей P(X,Y):")
    P_joint = calc_joint_prob(P_X, P_XY)
    print("Матрица P(X,Y):")
    for row in P_joint:
        print(f"{[round(float(p), 3) for p in row]}")

    # д) Энтропия на входе H(X)
    print("\nд) Энтропия на входе H(X):")
    H_X = calculate_entropy(P_X)
    print(f"H(X) = {round(H_X, 3)} бит")

    # е) Условная энтропия H(X|Y)
    print("\nе) Условная энтропия H(X|Y):")
    H_X_given_Y = calculate_conditional_entropy(P_XY, P_joint)
    print(f"H(X|Y) = {round(H_X_given_Y, 3)} бит")

    # ж) Количество информации при неполной достоверности I(X,Y)
    print("\nж) Количество информации при неполной достоверности I(X,Y):")
    I_XY = calculate_mutual_information(H_X, H_X_given_Y)
    print(f"I(X,Y) = {round(I_XY, 3)} бит")

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

    print(f"\nСреднее количество информации I(X,Y): {round(avg_I_XY, 3)} бит")

    print(f"\n{'=' * 60}")
    print("ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
