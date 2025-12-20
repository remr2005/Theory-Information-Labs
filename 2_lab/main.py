"""Main file - Лабораторная работа №2: Передача информации при наличии помех"""

import sys
from datetime import datetime

import numpy as np

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Preformatted, SimpleDocTemplate, Spacer

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

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
    result = []
    for p in arr:
        val = float(p)
        rounded = round(val, 5)
        if rounded == 0 and val != 0:
            result.append("0.00000>")
        else:
            # Форматируем в десятичном виде без научной нотации (фиксированный формат)
            result.append(f"{rounded:.5f}")
    return result


def format_matrix(matrix, max_elements=9, max_rows=7):
    """Форматирует матрицу без обрезания"""
    result = []
    for row in matrix:
        formatted_row = []
        for p in row:
            val = float(p)
            rounded = round(val, 5)
            if rounded == 0 and val != 0:
                formatted_row.append("0.00001>")
            else:
                # Форматируем в десятичном виде без научной нотации (фиксированный формат)
                formatted_row.append(f"{rounded:.5f}")
        result.append(formatted_row)
    return result


# Глобальные переменные для хранения результатов экспериментов
experiment_results = []


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


def save_to_pdf(output_text: str, filename: str = "lab2_output.pdf") -> None:
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
        left_margin = 1
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


def save_to_png(output_text: str, filename: str = "lab2_output.png") -> None:
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

    # б) Генерация матрицы переходных вероятностей P(X|Y)
    print("\nб) Генерация матрицы переходных вероятностей P(X|Y):")
    P_XY = make_matrix_transition(n)
    print("Матрица P(X|Y):")
    formatted_P_XY = format_matrix(P_XY)
    for i, row in enumerate(formatted_P_XY, 1):
        print(f"{i}: {' '.join(str(x) for x in row)}")

    # в) Расчет вероятностей P(Y) на выходе
    print("\nв) Расчет вероятностей P(Y) на выходе:")
    P_Y = calc_p_y(P_X, P_XY)
    formatted_P_Y = format_array(P_Y)
    print(f"P(Y) = {formatted_P_Y}")

    # г) Расчет матрицы совместных вероятностей P(X,Y)
    print("\nг) Расчет матрицы совместных вероятностей P(X,Y):")
    P_joint = calc_joint_prob(P_X, P_XY)
    print("Матрица P(X,Y):")
    formatted_P_joint = format_matrix(P_joint)
    for i, row in enumerate(formatted_P_joint, 1):
        print(f"{i}: {' '.join(str(x) for x in row)}")

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
    # Перехватываем вывод
    output_capture = OutputCapture()
    sys.stdout = output_capture

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

    # Восстанавливаем stdout
    sys.stdout = output_capture.original_stdout

    # Сохраняем в PDF
    output_text = output_capture.get_output()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"lab2_output_{timestamp}.pdf"
    save_to_pdf(output_text, pdf_filename)

    # Также предлагаем сохранить в PNG
    try:
        png_filename = f"lab2_output_{timestamp}.png"
        save_to_png(output_text, png_filename)
    except Exception as e:
        print(f"Не удалось сохранить PNG: {e}")


if __name__ == "__main__":
    main()
