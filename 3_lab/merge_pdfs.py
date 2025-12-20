"""Скрипт для вставки страниц из одного PDF в другой"""

from pypdf import PdfReader, PdfWriter


def merge_pdfs():
    """Вставляет страницы из lab3_output между 4 и 5 страницей основного PDF"""
    
    # Пути к файлам
    main_pdf_path = "3_lab/ПИ-б-о-231(2) Аметов К.Л. ТИиК ЛР3.pdf"
    insert_pdf_path = "3_lab/lab3_output_20251220_211652.pdf"
    output_pdf_path = "3_lab/ПИ-б-о-231(2) Аметов К.Л. ТИиК ЛР3_merged.pdf"
    
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
    
    # Добавляем первые 4 страницы из основного PDF
    for i in range(min(4, main_num_pages)):
        writer.add_page(main_reader.pages[i])
        print(f"Добавлена страница {i+1} из основного PDF")
    
    # Вставляем все страницы из PDF для вставки
    for i in range(insert_num_pages):
        writer.add_page(insert_reader.pages[i])
        print(f"Вставлена страница {i+1} из PDF для вставки")
    
    # Добавляем оставшиеся страницы из основного PDF (начиная с 5-й, индекс 4)
    for i in range(4, main_num_pages):
        writer.add_page(main_reader.pages[i])
        print(f"Добавлена страница {i+1} из основного PDF")
    
    # Сохраняем результат
    with open(output_pdf_path, "wb") as output_file:
        writer.write(output_file)
    
    print(f"\nГотово! Объединенный PDF сохранен в: {output_pdf_path}")
    print(f"Итого страниц в новом PDF: {len(writer.pages)}")


if __name__ == "__main__":
    merge_pdfs()

