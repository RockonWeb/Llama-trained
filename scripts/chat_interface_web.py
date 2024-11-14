import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st

# Функция для загрузки модели и токенизатора
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

# Функция для генерации ответа
def generate_answer(tokenizer, model, question, temperature=0.8, top_k=35, top_p=0.7, max_new_tokens=2048):
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
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Декодирование ответа
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_output[len(prompt):].strip()
    
    return answer

# Основная функция Streamlit для чата
def chat_interface():
    """
    Интерфейс чата с использованием Streamlit.
    """
    # Загрузка модели и токенизатора
    model_path = "C:/AI/models/fine_tuned_model"  # Обновите путь к вашей модели
    st.title("🤖 AI Чат")

    # Поля для ввода вопроса и настройки параметров
    question = st.text_input("Ваш вопрос:", "")
    st.sidebar.header("Настройки генерации")
    temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.8, step=0.1)
    top_k = st.sidebar.slider("Top-k", 1, 100, 35, step=1)
    top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 0.7, step=0.05)
    max_new_tokens = st.sidebar.number_input("Max New Tokens", 10, 4096, 2048, step=10)
    generate_button = st.button("Отправить")

    # Инициализация состояния для хранения истории ответов
    if "response" not in st.session_state:
        st.session_state["response"] = []

    if generate_button and question:
        # Загрузка токенизатора и модели, если они ещё не загружены
        if "tokenizer" not in st.session_state or "model" not in st.session_state:
            with st.spinner("Загрузка модели..."):
                try:
                    st.session_state["tokenizer"], st.session_state["model"] = load_model(model_path)
                    st.success("Модель загружена!")
                except Exception as e:
                    st.error(f"Ошибка при загрузке модели: {e}")
                    return

        # Генерация ответа
        with st.spinner("Генерация ответа..."):
            try:
                answer = generate_answer(
                    st.session_state["tokenizer"], 
                    st.session_state["model"], 
                    question, 
                    temperature=temperature, 
                    top_k=top_k, 
                    top_p=top_p, 
                    max_new_tokens=max_new_tokens
                )
                st.session_state["response"].append((question, answer))
                st.success("Ответ сгенерирован!")
            except Exception as e:
                st.error(f"Произошла ошибка при генерации ответа: {e}")

    # Отображение истории вопросов и ответов
    for i, (q, a) in enumerate(st.session_state["response"], 1):
        st.markdown(f"#### Вопрос {i}")
        st.write(f"{q}")
        st.markdown(f"#### Ответ {i}")
        st.markdown(
            f"<div style='background-color: #5b5b5b; padding: 10px; border-radius: 5px;'>{a}</div>",
            unsafe_allow_html=True
        )
        # Кнопка для копирования ответа
        st.button("Скопировать ответ", key=f"copy_button_{i}", on_click=lambda text=a: st.write(f"`{text}`"))

        st.write("---")

# Запуск интерфейса
if __name__ == "__main__":
    chat_interface()