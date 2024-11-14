def split_text(file_path, max_length=512):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 < max_length:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

if __name__ == "__main__":
    input_file = 'C:/AI/data/combined_lectures.txt'
    output_file = 'C:/AI/data/chunks.txt'
    chunks = split_text(input_file, max_length=512)
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + "\n\n=== END OF CHUNK ===\n\n")
    print(f"Текст разделён на {len(chunks)} частей и сохранён в {output_file}")
