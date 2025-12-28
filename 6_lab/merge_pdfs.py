"""Скрипт для объединения PDF файлов"""

from pypdf import PdfReader, PdfWriter


def merge_pdfs():
    """Добавляет первую страницу из FIRST_PAGE.pdf в начало 1_merged.pdf"""

    # Пути к файлам
    first_page_path = "6_lab/FIRST_PAGE.pdf"
    merged_pdf_path = "6_lab/1_merged.pdf"
    output_pdf_path = "6_lab/ПИ-б-о-231(2) Аметов К.Л. ЛР6.pdf"

    # Читаем первую страницу
    first_page_reader = PdfReader(first_page_path)
    first_page_num = len(first_page_reader.pages)
    print(f"FIRST_PAGE.pdf содержит {first_page_num} страниц")
    
    if first_page_num == 0:
        print("Ошибка: FIRST_PAGE.pdf не содержит страниц!")
        return

    # Читаем объединенный PDF
    merged_reader = PdfReader(merged_pdf_path)
    merged_num_pages = len(merged_reader.pages)
    print(f"1_merged.pdf содержит {merged_num_pages} страниц")

    # Создаем writer для нового PDF
    writer = PdfWriter()

    # Добавляем первую страницу из FIRST_PAGE.pdf
    writer.add_page(first_page_reader.pages[0])
    print(f"Добавлена первая страница из FIRST_PAGE.pdf")

    # Добавляем все страницы из объединенного PDF
    for i in range(merged_num_pages):
        writer.add_page(merged_reader.pages[i])
        print(f"Добавлена страница {i + 1} из 1_merged.pdf")

    # Сохраняем результат
    with open(output_pdf_path, "wb") as output_file:
        writer.write(output_file)

    print(f"\nГотово! Объединенный PDF сохранен в: {output_pdf_path}")
    print(f"Итого страниц в новом PDF: {len(writer.pages)}")
    print(f"Структура: 1 страница (титульная) + {merged_num_pages} страниц из 1_merged.pdf")


if __name__ == "__main__":
    merge_pdfs()

