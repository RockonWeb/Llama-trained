## **Requirements:**
- Python v3.10+ 

- Nvidia GPU
    
## Описание(u can see eng version further)
Обучение и использование языковой модели на основе PDF-файлов лекций.

## Структура проекта
- `data/` - данные и датасеты
- `scripts/` - скрипты для обработки данных и обучения
- `models/` - сохранённые модели и результаты обучения

## Установка
1. **CUDA и cuDNN (для пользователей с NVIDIA GPU)**
    *Примечание:* Если у вас нет NVIDIA GPU или вы планируете использовать CPU, этот шаг можно пропустить, но обучение будет значительно медленнее.

    Установите CUDA: https://developer.nvidia.com/cuda-downloads

    Установите cuDNN: https://developer.nvidia.com/cudnn
2. **Установите PyTorch с поддержкой CUDA:**
   Замените cu113 на вашу версию CUDA (например, cu116 для CUDA 11.6).
   ```
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
   ```

    Если вы используете CPU:
    
    ```
    pip install torch torchvision torchaudio
    ```
3. **Установите необходимые библиотеки:**
    ```
    pip install transformers datasets PyPDF2 langchain faiss-cpu
    ```
    *Примечание:* Если планируете использовать GPU для FAISS, установите faiss-gpu вместо faiss-cpu:
    ```
    pip install faiss-gpu
    ```
5. **Установите Miniconda(или Anaconda)** https://www.anaconda.com/download
7. **Создайте и активируйте виртуальное окружение:**
   ```
   conda create -n llm_env python=3.8
   conda activate llm_env
   ```

## Использование
1. **Извлечение текста из PDF:**
    ```
    python scripts/extract_text.py
    ```
2. **Разделение текста на фрагменты:**
    ```bash
    python scripts/split_text.py
3. **Создание датасета:**
    ```
    python scripts/create_dataset.py
    ```
4. **Обучение модели:**
    ```
    python scripts/train_model.py
    ```
5. **Начало общения с моделью:**
    ```
    python scripts/chat_interface.py 
    ```

---
---

## Description
Training and using a language model based on lecture PDF files.

## Project Structure
- `data/` - data and datasets
- `scripts/` - scripts for data processing and training
- `models/` - saved models and training results

## Installation
1. **CUDA and cuDNN (for NVIDIA GPU users)**  
   *Note:* If you don't have an NVIDIA GPU or plan to use a CPU, you can skip this step, but training will be significantly slower.

   Install CUDA: https://developer.nvidia.com/cuda-downloads  
   Install cuDNN: https://developer.nvidia.com/cudnn

2. **Install PyTorch with CUDA support:**  
   Replace `cu113` with your CUDA version (e.g., `cu116` for CUDA 11.6).
   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
   ```
   If you are using a CPU:
   ```bash
   pip install torch torchvision torchaudio
   ```

3. **Install the required libraries:**
   ```bash
   pip install transformers datasets PyPDF2 langchain faiss-cpu
   ```
   *Note:* If you plan to use GPU for FAISS, install `faiss-gpu` instead of `faiss-cpu`:
   ```bash
   pip install faiss-gpu
   ```

4. **Install Miniconda (or Anaconda)** https://www.anaconda.com/download

5. **Create and activate a virtual environment:**
   ```bash
   conda create -n llm_env python=3.8
   conda activate llm_env
   ```

## Usage
1. **Extract text from PDFs:**
   ```bash
   python scripts/extract_text.py
   ```
2. **Split text into chunks:**
   ```bash
   python scripts/split_text.py
   ```
3. **Create the dataset:**
   ```bash
   python scripts/create_dataset.py
   ```
4. **Train the model:**
   ```bash
   python scripts/train_model.py
   ```
5. **Start interacting with the model:**
   ```bash
   python scripts/chat_interface.py
   ```
