"""Объединенный файл для лабораторной работы №4: Коды Хэмминга"""

import sys
from datetime import datetime
from random import randint

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Preformatted, SimpleDocTemplate, Spacer

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

MSG_LEN = 61
ITER_COUNT = 6


def identity_matrix(size: int):
    """Создает единичную матрицу заданного размера"""
    res = []
    for i in range(size):
        row = [0] * size
        row[i] = 1
        res.append(row)
    return res


def transpose(matrix: list[list[int]]):
    """Транспонирует матрицу"""
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]


def print_combined(matrix_a, matrix_b):
    """Выводит две матрицы, объединенные по горизонтали"""
    for row_a, row_b in zip(matrix_a, matrix_b):
        print(*row_a, *row_b)


def write_combined(filename: str, matrix_a, matrix_b):
    """Записывает две матрицы в файл, объединенные по горизонтали"""
    with open(filename, "w") as f:
        for row_a, row_b in zip(matrix_a, matrix_b):
            f.write(" ".join(map(str, [*row_a, *row_b])) + "\n")


class OutputCapture:
    """Класс для перехвата вывода print и сохранения в строку"""

    def __init__(self):
        self.output = []
        self.original_stdout = sys.stdout

    def write(self, text):
        """Перехватывает вывод"""
        # Сохраняем весь текст, включая переносы строк
        self.output.append(text)
        self.original_stdout.write(text)  # Также выводим в консоль

    def flush(self):
        """Очистка буфера"""
        self.original_stdout.flush()

    def get_output(self):
        """Возвращает весь перехваченный вывод"""
        return "".join(self.output)


def save_to_pdf(output_text: str, filename: str = "lab4_output.pdf") -> None:
    """Сохраняет текст в PDF файл"""
    if not REPORTLAB_AVAILABLE:
        print("\nДля сохранения в PDF требуется библиотека reportlab.")
        print("Установите: pip install reportlab")
        return

    try:
        # Пытаемся загрузить шрифт с поддержкой кириллицы
        font_registered = False
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/System/Library/Fonts/Menlo.ttc",  # macOS
        ]

        for font_path in font_paths:
            try:
                pdfmetrics.registerFont(TTFont("CyrillicMono", font_path))
                font_registered = True
                break
            except (OSError, IOError, Exception):
                continue

        # Настраиваем отступы: уменьшаем слева, сверху, снизу; добавляем справа
        left_margin = 0
        right_margin = 200
        top_margin = 1
        bottom_margin = 1

        doc = SimpleDocTemplate(
            filename,
            pagesize=A4,
            leftMargin=left_margin,
            rightMargin=right_margin,
            topMargin=top_margin,
            bottomMargin=bottom_margin,
        )
        story = []
        styles = getSampleStyleSheet()

        # Создаем стиль с поддержкой кириллицы
        if font_registered:
            cyrillic_style = ParagraphStyle(
                "CyrillicCode",
                parent=styles["Code"],
                fontName="CyrillicMono",
                fontSize=8,
            )
        else:
            # Если не удалось загрузить шрифт, используем стандартный
            cyrillic_style = styles["Code"]

        # Используем Preformatted для сохранения форматирования
        # Разбиваем на строки и добавляем каждую строку
        lines = output_text.split("\n")
        for line in lines:
            if line.strip():
                # Используем моноширинный шрифт для матриц с ограничением ширины
                p = Preformatted(line, cyrillic_style, maxLineLength=115)
                story.append(p)
            else:
                story.append(Spacer(1, 6))  # Небольшой отступ для пустых строк

        doc.build(story)
        print(f"\nВывод сохранен в файл: {filename}")
    except Exception as e:
        print(f"\nОшибка при сохранении PDF: {e}")


def save_to_png(output_text: str, filename: str = "lab4_output.png") -> None:
    """Сохраняет текст в PNG файл (используя PIL)"""
    try:
        import textwrap

        from PIL import Image, ImageDraw, ImageFont

        # Параметры изображения
        font_size = 10
        line_height = 15
        padding = 20

        # Пытаемся загрузить шрифт сначала, чтобы измерить текст
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size
            )
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

        # Разбиваем текст на строки (каждая строка из print становится отдельной)
        lines = output_text.split("\n")

        # Измеряем ширину каждой строки и находим максимальную
        # Создаем временный draw для измерения
        temp_img = Image.new("RGB", (100, 100), color="white")
        temp_draw = ImageDraw.Draw(temp_img)

        max_text_width = 0
        final_lines = []

        for line in lines:
            # Обрабатываем каждую строку, включая пустые
            if not line:
                # Пустая строка - сохраняем её
                final_lines.append("\n")
                continue

            # Измеряем реальную ширину текста
            bbox = temp_draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]

            # Максимальная ширина для переноса (например, 8000px)
            max_width_px = 3500

            if text_width > max_width_px:
                # Нужно перенести строку
                # Приблизительно вычисляем количество символов, которые помещаются
                avg_char_width = text_width / len(line) if line else 1
                chars_per_line = int(max_width_px / avg_char_width)

                # Переносим по словам/пробелам если возможно, иначе по символам
                if " " in line:
                    wrapped = textwrap.wrap(line, width=chars_per_line)
                else:
                    # Для строк без пробелов (например, матрицы) разбиваем по символам
                    wrapped = [
                        line[i : i + chars_per_line]
                        for i in range(0, len(line), chars_per_line)
                    ]

                final_lines.extend(wrapped)
                # Обновляем максимальную ширину
                for wline in wrapped:
                    wbbox = temp_draw.textbbox((0, 0), wline, font=font)
                    wwidth = wbbox[2] - wbbox[0]
                    max_text_width = max(max_text_width, wwidth)
            else:
                # Строка помещается - добавляем как есть
                final_lines.append(line)
                max_text_width = max(max_text_width, text_width)

        # Вычисляем размеры изображения
        img_width = int(max_text_width + 2 * padding)
        num_lines = len(final_lines)
        img_height = num_lines * line_height + 2 * padding

        # Создаем финальное изображение
        img = Image.new("RGB", (img_width, img_height), color="white")
        draw = ImageDraw.Draw(img)

        # Рисуем текст - каждая строка на новой линии
        y = padding
        for line in final_lines:
            draw.text((padding, y), line, fill="black", font=font)
            y += line_height

        # Сохраняем изображение
        img.save(filename)
        print(f"\nВывод сохранен в файл: {filename}")
    except ImportError:
        print(
            "\nДля сохранения в PNG требуется библиотека Pillow. Установите: pip install Pillow"
        )
        print("Используется сохранение в PDF вместо PNG.")
        save_to_pdf(output_text, filename.replace(".png", ".pdf"))
    except Exception as e:
        print(f"\nОшибка при сохранении PNG: {e}")


def receiver_process(msg: list[int], check_matrix: list[list[int]]):
    """Обрабатывает принятое сообщение: проверяет на ошибки и исправляет их"""
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

        print(
            "", f"Сообщение с ошибкой, ошибка в {idx + 1} разряде систематического кода"
        )
        msg[idx] = int(not msg[idx])
        print("", "Скорректированный системный код:")
        print("", *msg)


def main() -> None:
    """Основная функция для выполнения лабораторной работы"""
    # Перехватываем вывод
    output_capture = OutputCapture()
    sys.stdout = output_capture

    # Генерация матриц
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

    # Загружаем проверочную матрицу для использования в receiver
    check_matrix = []
    with open("check_matrix.txt", "r") as f:
        for line in f:
            check_matrix.append(list(map(int, line.split())))

    # Выполняем эксперименты автоматически
    for iter in range(1, ITER_COUNT + 1):
        print(f"\n=== Эксперимент #{iter} ===")
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

        # Случайно вносим ошибку с вероятностью 50%
        if randint(0, 100) >= 50:
            err_idx = randint(0, len(code) - 1)
            print("", f"Вносим ошибку в {err_idx + 1} разряд")
            code[err_idx] = int(not code[err_idx])

        print("", "Передача сообщения приемнику...")
        receiver_process(code, check_matrix)

    # Восстанавливаем stdout
    sys.stdout = output_capture.original_stdout

    # Сохраняем в PDF
    output_text = output_capture.get_output()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"lab4_output_{timestamp}.pdf"
    save_to_pdf(output_text, pdf_filename)

    # Также предлагаем сохранить в PNG
    try:
        png_filename = f"lab4_output_{timestamp}.png"
        save_to_png(output_text, png_filename)
    except Exception as e:
        print(f"Не удалось сохранить PNG: {e}")


if __name__ == "__main__":
    main()
