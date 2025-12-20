"""Скрипт для вставки страниц из одного PDF в другой"""

from pypdf import PdfReader, PdfWriter


def merge_pdfs():
    """Вставляет страницы из lab2_output между 3 и 4 страницей основного PDF"""
    
    # Пути к файлам
    main_pdf_path = "2_lab/ПИ-б-о-231(2) Аметов К.Л. ЛР2.pdf"
    insert_pdf_path = "lab2_output_20251220_203823.pdf"
    output_pdf_path = "2_lab/ПИ-б-о-231(2) Аметов К.Л. ЛР2_merged.pdf"
    
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
    
    # Добавляем первые 3 страницы из основного PDF
    for i in range(min(3, main_num_pages)):
        writer.add_page(main_reader.pages[i])
        print(f"Добавлена страница {i+1} из основного PDF")
    
    # Вставляем все страницы из PDF для вставки
    for i in range(insert_num_pages):
        writer.add_page(insert_reader.pages[i])
        print(f"Вставлена страница {i+1} из PDF для вставки")
    
    # Добавляем оставшиеся страницы из основного PDF (начиная с 4-й, индекс 3)
    for i in range(3, main_num_pages):
        writer.add_page(main_reader.pages[i])
        print(f"Добавлена страница {i+1} из основного PDF")
    
    # Сохраняем результат
    with open(output_pdf_path, "wb") as output_file:
        writer.write(output_file)
    
    print(f"\nГотово! Объединенный PDF сохранен в: {output_pdf_path}")
    print(f"Итого страниц в новом PDF: {len(writer.pages)}")


if __name__ == "__main__":
    merge_pdfs()

