import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict, concatenate_datasets
from peft import get_peft_model, LoraConfig, TaskType

def load_text_files_as_dataset(text_directory):
    texts = []
    for filename in os.listdir(text_directory):
        if filename.endswith(".txt"):
            with open(os.path.join(text_directory, filename), "r", encoding="utf-8") as file:
                texts.append({"text": file.read()})
    return Dataset.from_list(texts)

def load_augmented_data(augmented_path):
    augmented_texts = []
    for filename in os.listdir(augmented_path):
        if filename.endswith(".txt"):
            with open(os.path.join(augmented_path, filename), "r", encoding="utf-8") as file:
                augmented_texts.append({"text": file.read()})
    return Dataset.from_list(augmented_texts)

def main():
    # Проверка доступности GPU
    print("Проверка доступности GPU:")
    if torch.cuda.is_available():
        print("GPU доступен.")
        print(f"Количество GPU: {torch.cuda.device_count()}")
        print(f"Имя первого GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("GPU не доступен, будет использоваться CPU.")
        device = torch.device("cpu")

    # Загрузка данных из train, test и val папок
    dataset_base_path = 'C:/AI/data/dataset'
    dataset = DatasetDict({
        "train": load_text_files_as_dataset(os.path.join(dataset_base_path, "train")),
        "test": load_text_files_as_dataset(os.path.join(dataset_base_path, "test")),
        "val": load_text_files_as_dataset(os.path.join(dataset_base_path, "val"))
    })
    print("Датасет загружен с разделами: train, test и val.")

    # Загрузка аугментированного датасета
    augmented_path = 'C:/AI/data/augmented'
    try:
        augmented_dataset = load_augmented_data(augmented_path)
        print(f"Аугментированные данные загружены из {augmented_path}.")
        # Добавляем аугментированные данные в тренировочный набор
        dataset["train"] = concatenate_datasets([dataset["train"], augmented_dataset])
        print("Аугментированные данные добавлены в тренировочный набор.")
    except Exception as e:
        print(f"Ошибка при загрузке аугментированных данных из {augmented_path}: {e}")
        return

    # Загрузка токенизатора и модели LLaMA
    model_name = "meta-llama/Llama-3.2-3b"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Токенизатор '{model_name}' загружен.")
    except Exception as e:
        print(f"Ошибка при загрузке токенизатора '{model_name}': {e}")
        return

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_auth_token="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"
        )
        print(f"Модель '{model_name}' загружена.")
    except Exception as e:
        print(f"Ошибка при загрузке модели '{model_name}': {e}")
        return

    # Перемещение модели на устройство
    model.to(device)
    print(f"Модель перемещена на устройство: {device}")

    # Проверка и назначение pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"pad_token установлен как eos_token: {tokenizer.pad_token}")

    # Обновление конфигурации модели
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    print("Градиентный чекпоинтинг включён.")

    # Настройка LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)
    print("LoRA конфигурация применена.")

    # Токенизация данных с добавлением labels
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=1024
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        if tokenizer.pad_token_id is not None:
            tokenized["labels"] = [
                [label if label != tokenizer.pad_token_id else -100 for label in labels]
                for labels in tokenized["labels"]
            ]
        
        return tokenized

    # Применение токенизации к каждому разделу датасета
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    print("Данные токенизированы и метки добавлены.")

    # Форматирование датасета для PyTorch
    if "train" in tokenized_datasets:
        train_dataset = tokenized_datasets["train"].remove_columns(["text"])
    else:
        train_dataset = tokenized_datasets.remove_columns(["text"])

    train_dataset.set_format("torch")
    print("Набор данных отформатирован для PyTorch.")

    # Инициализация Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    print("Data Collator инициализирован.")

    # Настройка аргументов обучения
    training_args = TrainingArguments(
        output_dir="C:/AI/models/results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=True,
        logging_steps=200,
        logging_dir="C:/AI/models/logs",
    )
    print("Аргументы обучения настроены.")

    # Создание тренера
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    print("Trainer создан.")

    # Запуск обучения
    print("Начало обучения...")
    trainer.train()
    print("Обучение завершено.")

    # Сохранение модели и токенизатора
    save_directory = "C:/AI/models/fine_tuned_model"
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Обученная модель с LoRA адаптацией сохранена в {save_directory}")

if __name__ == "__main__":
    main()
