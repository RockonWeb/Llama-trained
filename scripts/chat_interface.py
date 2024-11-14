import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path):
    """
    Загрузка токенизатора и модели из указанного пути.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        torch_dtype=torch.float16
    )
    return tokenizer, model

def generate_answer(tokenizer, model, question, max_new_tokens=2048):
    """
    Генерация ответа на вопрос с использованием модели.
    """
    prompt = f"Отвечай на РУССКОМ. {question}<|endoftext|>"
    
    # Токенизация промпта
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    ).to(model.device)
    
    # Генерация ответа
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.7,
        top_k=35,
        temperature=0.8,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Декодирование ответа
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_output[len(prompt):].strip()
    
    return answer

def chat():
    """
    Интерфейс чата для взаимодействия с пользователем.
    """
    model_path = "C:/AI/models/fine_tuned_model"  # Обновите путь к вашей модели

    tokenizer, model = load_model(model_path)

    print("Модель загружена. Введите ваш вопрос (или 'выход' для завершения):")

    while True:
        question = input("Вы: ")
        if question.lower() in ["выход", "exit", "quit"]:
            print("Завершение сеанса.")
            break
        try:
            answer = generate_answer(tokenizer, model, question)
            print(f"Ответ: {answer}\n")
        except Exception as e:
            print(f"Произошла ошибка при генерации ответа: {e}\n")

if __name__ == "__main__":
    chat()
