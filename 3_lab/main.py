"""Main file - Лабораторная работа №3: Обобщенные характеристики сигналов и каналов"""

import sys
from datetime import datetime
from math import log2

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


def save_to_pdf(output_text: str, filename: str = "lab3_output.pdf") -> None:
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


def save_to_png(output_text: str, filename: str = "lab3_output.png") -> None:
    """Сохраняет текст в PNG файл (используя PIL)"""
    try:
        import textwrap

        from PIL import Image, ImageDraw, ImageFont

        # Параметры изображения
        font_size = 50
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


def format_array(arr, max_elements=9):
    """Форматирует массив без обрезания"""
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

    # б) Генерация массива длительностей Tx[i]
    print("\nб) Генерация массива длительностей Tx[i] (мкс):")
    Tx = make_array_durations(n)
    formatted_Tx = format_array(Tx)
    print(f"Tx = {formatted_Tx} мкс")

    # в) Генерация матрицы вероятностей ошибок P[X, Y]
    print("\nв) Генерация матрицы вероятностей ошибок P[X, Y]:")
    P_XY = make_matrix_error_probabilities(n)
    print("Матрица P[X, Y]:")
    formatted_P_XY = format_matrix(P_XY)
    for i, row in enumerate(formatted_P_XY, 1):
        print(f"{i}: {' '.join(str(x) for x in row)}")

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
    print("\nСкорость передачи I(x) = H(x) / тау:")
    print(f"I(x) = {H_x:.5f} бит / {tau:.5f} мкс = {format_speed(I_x)}")

    # Пропускная способность C = (log₂ N) / тау
    max_entropy = log2(n)
    C_noise_free = max_entropy / tau_seconds  # бит/с
    print("\nПропускная способность C = (log₂ N) / тау:")
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
    print("\nСкорость передачи I(Y, Z) = (H(x) - H(X/Y)) / тау:")
    print(
        f"I(Y, Z) = ({H_x:.5f} - {H_X_given_Y:.5f}) бит / {tau:.5f} мкс = {format_speed(I_YZ)}"
    )

    # Пропускная способность C = (log₂ N - H(X/Y)) / тау
    C_with_noise = (max_entropy - H_X_given_Y) / tau_seconds  # бит/с
    print("\nПропускная способность C = (log₂ N - H(X/Y)) / тау:")
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
    # Перехватываем вывод
    output_capture = OutputCapture()
    sys.stdout = output_capture

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
    print(f"\n2. Средняя скорость передачи без помех: {format_speed(avg_I_x)}")
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
    print("\n4. Средняя длительность символа влияет на скорость передачи:")
    avg_tau = np.mean([r["tau"] for r in experiment_results_noise_free])
    print(f"   Средняя длительность тау = {round(avg_tau, 5)} мкс")
    print("\n5. При увеличении количества символов N максимальная энтропия log₂(N)")
    print("   увеличивается, что потенциально позволяет передавать больше информации.")

    print(f"\n{'=' * 70}")
    print("ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    print(f"{'=' * 70}")

    # Восстанавливаем stdout
    sys.stdout = output_capture.original_stdout

    # Сохраняем в PDF
    output_text = output_capture.get_output()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"lab3_output_{timestamp}.pdf"
    save_to_pdf(output_text, pdf_filename)

    # Также предлагаем сохранить в PNG
    try:
        png_filename = f"lab3_output_{timestamp}.png"
        save_to_png(output_text, png_filename)
    except Exception as e:
        print(f"Не удалось сохранить PNG: {e}")


if __name__ == "__main__":
    main()
