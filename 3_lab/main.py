"""Main file - Лабораторная работа №3: Обобщенные характеристики сигналов и каналов"""

import numpy as np
from math import log2

from utils import (
    calculate_average_duration,
    calculate_conditional_entropy,
    calculate_conditional_probability_X_given_Y,
    calculate_entropy,
    calculate_joint_probability,
    calculate_output_probabilities,
    make_array_durations,
    make_array_probabilities,
    make_matrix_error_probabilities,
)

# Вариант 1: N = 8
N = 8

# Глобальные переменные для хранения результатов
experiment_results_noise_free = []
experiment_results_with_noise = []


def format_array(arr, max_elements=9):
    """Форматирует массив с обрезанием для больших размеров"""
    if len(arr) <= max_elements:
        return [round(float(p), 5) for p in arr]
    else:
        start = [round(float(p), 5) for p in arr[:4]]
        end = [round(float(p), 5) for p in arr[-4:]]
        return start + ["..."] + end


def format_matrix(matrix, max_elements=9, max_rows=7):
    """Форматирует матрицу с обрезанием для больших размеров"""
    if matrix.shape[0] > max_rows:
        rows_to_show = matrix[:3]
        rows_to_show = np.vstack([rows_to_show, matrix[-3:]])
    else:
        rows_to_show = matrix

    result = []
    for i, row in enumerate(rows_to_show):
        if matrix.shape[0] > max_rows and i == 3:
            result.append(
                ["..."] * (9 if matrix.shape[1] > max_elements else matrix.shape[1])
            )
        elif matrix.shape[1] <= max_elements:
            result.append([round(float(p), 5) for p in row])
        else:
            start = [round(float(p), 5) for p in row[:4]]
            end = [round(float(p), 5) for p in row[-4:]]
            result.append(start + ["..."] + end)

    return result


def format_speed(value: float) -> str:
    """
    Форматирует скорость передачи с правильными единицами измерения.
    Конвертирует бит/с в подходящие единицы (кбит/с, Мбит/с)
    """
    if value >= 1e6:
        return f"{value / 1e6:.4f} Мбит/с"
    elif value >= 1e3:
        return f"{value / 1e3:.4f} кбит/с"
    else:
        return f"{value:.4f} бит/с"


def conduct_experiment(experiment_num: int, n: int = N) -> dict:
    """
    Проведение одного эксперимента
    
    Args:
        experiment_num (int): Номер эксперимента
        n (int): Количество дискретных сообщений
        
    Returns:
        dict: Результаты эксперимента
    """
    print(f"\n{'=' * 70}")
    print(f"ЭКСПЕРИМЕНТ {experiment_num}")
    print(f"{'=' * 70}")

    # а) Генерация массива вероятностей Px[i] на входе
    print("\nа) Генерация массива вероятностей Px[i] на входе:")
    Px = make_array_probabilities(n)
    formatted_Px = format_array(Px)
    print(f"Px = {formatted_Px}")
    if len(Px) > 9:
        print("(массив обрезан для удобства отображения)")

    # б) Генерация массива длительностей Tx[i]
    print("\nб) Генерация массива длительностей Tx[i] (мкс):")
    Tx = make_array_durations(n)
    formatted_Tx = format_array(Tx)
    print(f"Tx = {formatted_Tx} мкс")
    if len(Tx) > 9:
        print("(массив обрезан для удобства отображения)")

    # в) Генерация матрицы вероятностей ошибок P[X, Y]
    print("\nв) Генерация матрицы вероятностей ошибок P[X, Y]:")
    P_XY = make_matrix_error_probabilities(n)
    print("Матрица P[X, Y]:")
    formatted_P_XY = format_matrix(P_XY)
    for row in formatted_P_XY:
        print(f"{row}")
    if n > 7:
        print("(матрица обрезана для удобства отображения)")

    # Расчет энтропии H(x)
    print("\nг) Расчет энтропии H(x):")
    H_x = calculate_entropy(Px)
    print(f"H(x) = {round(H_x, 5)} бит")

    # Расчет средней длительности тау
    print("\nд) Расчет средней длительности символа:")
    tau = calculate_average_duration(Px, Tx)
    print(f"тау = {round(tau, 5)} мкс")

    # КАНАЛ БЕЗ ПОМЕХ
    print(f"\n{'=' * 70}")
    print("КАНАЛ БЕЗ ПОМЕХ")
    print(f"{'=' * 70}")

    # Скорость передачи I(x) = H(x) / тау
    # Конвертируем тау из мкс в секунды: 1 мкс = 1e-6 с
    tau_seconds = tau * 1e-6
    I_x = H_x / tau_seconds  # бит/с
    print(f"\nСкорость передачи I(x) = H(x) / тау:")
    print(f"I(x) = {H_x:.5f} бит / {tau:.5f} мкс = {format_speed(I_x)}")

    # Пропускная способность C = (log₂ N) / тау
    max_entropy = log2(n)
    C_noise_free = max_entropy / tau_seconds  # бит/с
    print(f"\nПропускная способность C = (log₂ N) / тау:")
    print(f"C = {max_entropy:.5f} бит / {tau:.5f} мкс = {format_speed(C_noise_free)}")

    # КАНАЛ С ПОМЕХАМИ
    print(f"\n{'=' * 70}")
    print("КАНАЛ С ПОМЕХАМИ")
    print(f"{'=' * 70}")

    # Расчет матрицы совместных вероятностей P(X,Y)
    P_joint = calculate_joint_probability(Px, P_XY)
    
    # Расчет вероятностей на выходе P(Y)
    P_Y = calculate_output_probabilities(Px, P_XY)
    
    # Расчет условных вероятностей P(X|Y) из совместных вероятностей
    P_X_given_Y = calculate_conditional_probability_X_given_Y(P_joint, P_Y)

    # Расчет условной энтропии H(X/Y)
    print("\nРасчет условной энтропии H(X/Y):")
    H_X_given_Y = calculate_conditional_entropy(P_joint, P_X_given_Y)
    print(f"H(X/Y) = {round(H_X_given_Y, 5)} бит")

    # Скорость передачи I(Y, Z) = (H(x) - H(X/Y)) / тау
    I_YZ = (H_x - H_X_given_Y) / tau_seconds  # бит/с
    print(f"\nСкорость передачи I(Y, Z) = (H(x) - H(X/Y)) / тау:")
    print(
        f"I(Y, Z) = ({H_x:.5f} - {H_X_given_Y:.5f}) бит / {tau:.5f} мкс = {format_speed(I_YZ)}"
    )

    # Пропускная способность C = (log₂ N - H(X/Y)) / тау
    C_with_noise = (max_entropy - H_X_given_Y) / tau_seconds  # бит/с
    print(f"\nПропускная способность C = (log₂ N - H(X/Y)) / тау:")
    print(
        f"C = ({max_entropy:.5f} - {H_X_given_Y:.5f}) бит / {tau:.5f} мкс = {format_speed(C_with_noise)}"
    )

    # Сохранение результатов
    result_noise_free = {
        "experiment_num": experiment_num,
        "Px": Px,
        "Tx": Tx,
        "H_x": H_x,
        "tau": tau,
        "I_x": I_x,
        "C": C_noise_free,
    }

    result_with_noise = {
        "experiment_num": experiment_num,
        "Px": Px,
        "Tx": Tx,
        "P_XY": P_XY,
        "H_x": H_x,
        "H_X_given_Y": H_X_given_Y,
        "tau": tau,
        "I_YZ": I_YZ,
        "C": C_with_noise,
    }

    return result_noise_free, result_with_noise


def main() -> None:
    """
    Основная функция для проведения экспериментов
    """
    print("ЛАБОРАТОРНАЯ РАБОТА №3")
    print("ОБОБЩЕННЫЕ ХАРАКТЕРИСТИКИ СИГНАЛОВ И КАНАЛОВ")
    print(f"Вариант 1: N = {N}")
    print("=" * 70)

    # Задание I: Проведение комплекса численных экспериментов (не менее 6)
    num_experiments = 6
    print(f"\nЗАДАНИЕ I: Проведение {num_experiments} экспериментов")
    print("=" * 70)

    for i in range(num_experiments):
        result_noise_free, result_with_noise = conduct_experiment(i + 1)
        experiment_results_noise_free.append(result_noise_free)
        experiment_results_with_noise.append(result_with_noise)

    # Задание II: Расчет средних значений
    print(f"\n{'=' * 70}")
    print("ЗАДАНИЕ II: СРЕДНИЕ ЗНАЧЕНИЯ")
    print(f"{'=' * 70}")

    # Канал без помех
    print("\nКАНАЛ БЕЗ ПОМЕХ:")
    avg_I_x = np.mean([r["I_x"] for r in experiment_results_noise_free])
    avg_C_noise_free = np.mean([r["C"] for r in experiment_results_noise_free])
    print(f"Средняя скорость передачи I(x): {format_speed(avg_I_x)}")
    print(f"Средняя пропускная способность C: {format_speed(avg_C_noise_free)}")

    # Канал с помехами
    print("\nКАНАЛ С ПОМЕХАМИ:")
    avg_I_YZ = np.mean([r["I_YZ"] for r in experiment_results_with_noise])
    avg_C_with_noise = np.mean([r["C"] for r in experiment_results_with_noise])
    print(f"Средняя скорость передачи I(Y, Z): {format_speed(avg_I_YZ)}")
    print(f"Средняя пропускная способность C: {format_speed(avg_C_with_noise)}")

    # Задание III: Выводы
    print(f"\n{'=' * 70}")
    print("ЗАДАНИЕ III: ВЫВОДЫ")
    print(f"{'=' * 70}")
    print(
        "\n1. Скорость передачи в канале с помехами всегда меньше, чем в канале без помех,"
    )
    print("   так как часть информации теряется из-за действия помех.")
    print(
        f"\n2. Средняя скорость передачи без помех: {format_speed(avg_I_x)}"
    )
    print(f"   Средняя скорость передачи с помехами: {format_speed(avg_I_YZ)}")
    print(
        f"   Разница: {format_speed(avg_I_x - avg_I_YZ)} (потеря {((avg_I_x - avg_I_YZ) / avg_I_x * 100):.2f}%)"
    )
    print(
        f"\n3. Пропускная способность канала без помех: {format_speed(avg_C_noise_free)}"
    )
    print(
        f"   Пропускная способность канала с помехами: {format_speed(avg_C_with_noise)}"
    )
    print(
        f"   Разница: {format_speed(avg_C_noise_free - avg_C_with_noise)} (потеря {((avg_C_noise_free - avg_C_with_noise) / avg_C_noise_free * 100):.2f}%)"
    )
    print(
        "\n4. Средняя длительность символа влияет на скорость передачи:"
    )
    avg_tau = np.mean([r["tau"] for r in experiment_results_noise_free])
    print(f"   Средняя длительность тау = {round(avg_tau, 5)} мкс")
    print(
        "\n5. При увеличении количества символов N максимальная энтропия log₂(N)"
    )
    print("   увеличивается, что потенциально позволяет передавать больше информации.")

    print(f"\n{'=' * 70}")
    print("ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

