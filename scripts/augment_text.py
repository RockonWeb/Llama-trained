import os
import nlpaug.augmenter.word as naw

def augment_text_files(text_directory, output_directory, augmentations_per_file=2):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Использование альтернативного аугментатора, например, swap слов
    augmenter = naw.RandomWordAug(action="swap") 

    for filename in os.listdir(text_directory):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(text_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                original_text = f.read()

            for i in range(augmentations_per_file):
                augmented_text = augmenter.augment(original_text)
                
                # Проверка, если результат - список, объединяем в строку
                if isinstance(augmented_text, list):
                    augmented_text = ' '.join(augmented_text)

                augmented_filename = f"{os.path.splitext(filename)[0]}_aug_{i + 1}.txt"
                augmented_path = os.path.join(output_directory, augmented_filename)
                
                with open(augmented_path, 'w', encoding='utf-8') as f:
                    f.write(augmented_text)
                print(f"Аугментированный файл сохранен как '{augmented_filename}'")

# Пример использования:
text_directory = 'C:/AI/data/text'
output_directory = 'C:/AI/data/augmented'
augmentations_per_file = 5  # Количество аугментаций для каждого исходного файла
augment_text_files(text_directory, output_directory, augmentations_per_file)