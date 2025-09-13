"""Main file"""

from math import log2

from utils import calculate_entropy, make_array


def step() -> None:
    """
    Функция для выполнения эксперимента
    """

    array = make_array(8)
    print("Массив вероятностей появления совокупности дискретных сообщений:")
    print(" ".join([str(i) for i in array]))
    print(f"Кол-во информации I(X) = {calculate_entropy(array)}")


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


if __name__ == "__main__":
    main()
