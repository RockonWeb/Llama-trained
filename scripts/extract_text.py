import os
from pdfminer.high_level import extract_text

def extract_text_from_pdfs(pdf_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(pdf_directory):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            try:
                text = extract_text(pdf_path, codec='utf-8')
                output_filename = os.path.splitext(filename)[0] + '.txt'
                output_path = os.path.join(output_directory, output_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Текст из '{filename}' сохранен в '{output_filename}'")
            except Exception as e:
                print(f"Ошибка при обработке файла '{filename}': {e}")

# Пример использования:
pdf_directory = 'C:/AI/data/pdfs'
output_directory = 'C:/AI/data/text'
extract_text_from_pdfs(pdf_directory, output_directory)