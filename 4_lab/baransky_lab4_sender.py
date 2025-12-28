import subprocess
from random import randint

MSG_LEN = 53
DEFAULT_ITER_COUNT = 6


def identity_matrix(size: int):
    res = []
    for i in range(size):
        row = [0] * size
        row[i] = 1
        res.append(row)

    return res


def transpose(matrix: list[list[int]]):
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]


def print_combined(matrix_a, matrix_b):
    for row_a, row_b in zip(matrix_a, matrix_b):
        print(*row_a, *row_b)


def write_combined(filename: str, matrix_a, matrix_b):
    with open(filename, "w") as f:
        for row_a, row_b in zip(matrix_a, matrix_b):
            f.write(" ".join(map(str, [*row_a, *row_b])) + "\n")


n = 0
while 2**MSG_LEN > (2**n / (n + 1)):
    n += 1

p = n - MSG_LEN
u_matrix = identity_matrix(MSG_LEN)
h_p_matrix = []
i = 3
for _ in range(MSG_LEN):
    bin_i = bin(i)
    if bin_i.count("1") == 1:
        i += 1
        bin_i = bin(i)

    h_p_matrix.append(list(map(int, bin_i[2:].zfill(p))))
    i += 1

print(f"Производящая матрица P (k = {MSG_LEN}, n = {n}, p = {p}):")
print_combined(u_matrix, h_p_matrix)

h_t_matrix = transpose(h_p_matrix)
id_matrix = identity_matrix(p)
print("\nПроверочная матрица H:")
print_combined(h_t_matrix, id_matrix)
write_combined("check_matrix.txt", h_t_matrix, id_matrix)

iter_count = None
print("Введите количество экспериментов (или оставьте значение по умолчанию)")
while True:
    inp = input(f"[{DEFAULT_ITER_COUNT}] >>> ")
    if inp == "":
        iter_count = DEFAULT_ITER_COUNT
        break
    try:
        iter_count = int(inp)
        if iter_count > 0:
            break
    except ValueError:
        pass

for iter in range(1, iter_count + 1):
    print(f"=== Эксперимент #{iter} ===")
    seq = list(map(int, bin(randint(0, 2**MSG_LEN - 1))[2:].zfill(MSG_LEN)))
    chksums = []
    for row in h_t_matrix:
        chksum = 0
        for j in range(len(row)):
            if row[j] == 1:
                chksum ^= seq[j]

        chksums.append(chksum)

    print("", "Сгенерированная комбинация:")
    print("", *seq)
    print("", "Систематический код:")
    code = seq + chksums
    print("", *code)
    if randint(0, 100) >= 50:
        err_idx = randint(0, len(code) - 1)
        print("", f"Вносим ошибку в {err_idx} разряд")
        code[err_idx] = int(not code[err_idx])

    print("", "Передача сообщения приемнику...")
    subprocess.run(
        ["python", "baransky_lab4_receiver.py"],
        input=" ".join(map(str, code)),
        text=True,
    )
    print()
