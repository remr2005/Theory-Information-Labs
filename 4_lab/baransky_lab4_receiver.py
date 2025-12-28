check_matrix = []
with open("check_matrix.txt", "r") as f:
    for line in f:
        check_matrix.append(list(map(int, line.split())))

msg = list(map(int, input().split()))
print("", "Принято сообщение:")
print("", *msg)

chksums = []
for row in check_matrix:
    chksum = 0
    for j in range(len(row)):
        if row[j] == 1:
            chksum ^= msg[j]

    chksums.append(chksum)

print("", "Синдром ошибки: (", *chksums, ")")
if chksums.count(1) == 0:
    print("", "Сообщение не содержит ошибок!")
else:
    idx = None
    for j in range(len(check_matrix[0])):
        col = [check_matrix[i][j] for i in range(len(check_matrix))]
        if col == chksums:
            idx = j
            break

    print("", f"Сообщение с ошибкой, ошибка в {idx} разряде систематического кода")
    msg[idx] = int(not msg[idx])
    print("", "Скорректированный системный код:")
    print("", *msg)
