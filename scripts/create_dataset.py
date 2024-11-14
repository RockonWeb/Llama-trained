import os
from sklearn.model_selection import train_test_split

def create_dataset(text_directory, output_directory, test_size=0.2, val_size=0.1):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    texts = []
    filenames = []
    for filename in os.listdir(text_directory):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(text_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                filenames.append(filename)

    # Разделение на временную и тестовую выборки
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, filenames, test_size=test_size, random_state=42
    )

    # Относительный размер валидационной выборки
    val_relative_size = val_size / (1 - test_size)

    # Разделение на обучающую и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_size, random_state=42
    )

    # Функция для сохранения данных
    def save_data(split_texts, split_filenames, split_name):
        split_dir = os.path.join(output_directory, split_name)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        for text, filename in zip(split_texts, split_filenames):
            with open(os.path.join(split_dir, filename), 'w', encoding='utf-8') as f:
                f.write(text)

    # Сохранение данных
    save_data(X_train, y_train, 'train')
    save_data(X_val, y_val, 'val')
    save_data(X_test, y_test, 'test')

    print("Датасет успешно создан и разделен на обучающую, валидационную и тестовую выборки.")

# Пример использования:
text_directory = 'C:/AI/data/augmented'
output_directory = 'C:/AI/data/dataset'
create_dataset(text_directory, output_directory)