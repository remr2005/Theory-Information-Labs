"""Main file"""

from math import log2

from utils import calculate_entropy, make_array

I_array = []


def step() -> None:
    """
    Функция для выполнения эксперимента
    """
    global I_array
    array = make_array(8)
    I_array.append(calculate_entropy(array))
    print("Массив вероятностей появления совокупности дискретных сообщений:")
    print(" ".join([str(round(i, 5)) for i in array]))
    print(f"Кол-во информации I(X) = {round(calculate_entropy(array), 5)}")


def main() -> None:
    """
    Main function
    """
    print(f"Максимальная энтропия {log2(8)}")

    for i in range(int(input("Сколько экспериментов нужно провести?"))):
        print()
        print(f"Эксперимент {i + 1}")
        step()
        print()
    print(f"Среднее количество информации {round(sum(I_array) / len(I_array), 5)}")


if __name__ == "__main__":
    main()
