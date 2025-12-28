"""Скрипт для вставки страниц из одного PDF в другой"""

from pypdf import PdfReader, PdfWriter


def merge_pdfs():
    """Удаляет страницы 4, 5, 6 и вставляет вместо них lab4_output"""

    # Пути к файлам
    main_pdf_path = "4_lab/1.pdf"
    insert_pdf_path = "4_lab/lab4_output_20251220_221000.pdf"
    output_pdf_path = "4_lab/ПИ-б-о-231(2) Аметов К.Л. ТИиК ЛР4.pdf"

    # Читаем основной PDF
    main_reader = PdfReader(main_pdf_path)
    main_num_pages = len(main_reader.pages)
    print(f"Основной PDF содержит {main_num_pages} страниц")

    # Читаем PDF для вставки
    insert_reader = PdfReader(insert_pdf_path)
    insert_num_pages = len(insert_reader.pages)
    print(f"PDF для вставки содержит {insert_num_pages} страниц")

    # Создаем writer для нового PDF
    writer = PdfWriter()

    # Добавляем первые 3 страницы из основного PDF (страницы 1-3)
    for i in range(min(3, main_num_pages)):
        writer.add_page(main_reader.pages[i])
        print(f"Добавлена страница {i + 1} из основного PDF")

    # Вставляем все страницы из PDF для вставки (вместо страниц 4, 5, 6)
    for i in range(insert_num_pages):
        writer.add_page(insert_reader.pages[i])
        print(f"Вставлена страница {i + 1} из PDF для вставки (заменяет страницы 4-6)")

    # Добавляем оставшиеся страницы из основного PDF (начиная с 7-й, индекс 6)
    # Пропускаем страницы 4, 5, 6 (индексы 3, 4, 5)
    for i in range(6, main_num_pages):
        writer.add_page(main_reader.pages[i])
        print(f"Добавлена страница {i + 1} из основного PDF")

    # Сохраняем результат
    with open(output_pdf_path, "wb") as output_file:
        writer.write(output_file)

    print(f"\nГотово! Объединенный PDF сохранен в: {output_pdf_path}")
    print(f"Итого страниц в новом PDF: {len(writer.pages)}")
    print(
        f"Удалены страницы 4, 5, 6 из основного PDF, заменены на {insert_num_pages} страниц из PDF результатов"
    )


if __name__ == "__main__":
    merge_pdfs()
