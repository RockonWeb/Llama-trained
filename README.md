# My LLM Project

## Описание
Обучение и использование языковой модели на основе PDF-файлов лекций.

## Структура проекта
- `data/` - данные и датасеты
- `scripts/` - скрипты для обработки данных и обучения
- `models/` - сохранённые модели и результаты обучения

## Установка
1. CUDA и cuDNN (для пользователей с NVIDIA GPU)
    Примечание: Если у вас нет NVIDIA GPU или вы планируете использовать CPU, этот шаг можно пропустить, но обучение будет значительно медленнее.

    Установите CUDA: https://developer.nvidia.com/cuda-downloads

    Установите cuDNN: https://developer.nvidia.com/cudnn
2. Установите PyTorch с поддержкой CUDA:
    Замените cu113 на вашу версию CUDA (например, cu116 для CUDA 11.6).
        ```bash
        Копировать код
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

    Если вы используете CPU:
        ```bash
        Копировать код
        pip install torch torchvision torchaudio
3. Установите необходимые библиотеки:
        ```bash
        pip install transformers datasets PyPDF2 langchain faiss-cpu

    Примечание: Если планируете использовать GPU для FAISS, установите faiss-gpu вместо faiss-cpu:
        ```bash
        pip install faiss-gpu
4. Установите Miniconda(или Anaconda)
5. Создайте и активируйте виртуальное окружение:
   ```bash
   conda create -n llm_env python=3.8
   conda activate llm_env

## Использование
1. Извлечение текста из PDF:
    ```bash
    python scripts/extract_text.py
2. Разделение текста на фрагменты:
    ```bash
    python scripts/split_text.py
3. Создание датасета:
    ```bash
    python scripts/create_dataset.py
4. Обучение модели:
    ```bash
    python scripts/train_model.py
5. Начало общения с моделью:
    ```bash
    python scripts/chat_interface.py