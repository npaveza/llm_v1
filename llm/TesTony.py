import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
import re

# Cargar el modelo GPT-2 preentrenado de Tony
model_name = 'DeepESP/gpt2-spanish'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Añadir un token de relleno (pad token) si es necesario
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Inicializar la memoria
memory_file = "tony_memory.json"
if os.path.exists(memory_file):
    with open(memory_file, "r") as f:
        memory = json.load(f)
else:
    memory = []

# Función para eliminar la redundancia en la respuesta generada
def remove_redundancy(response):
    """Eliminar la redundancia en la respuesta generada."""
    # Dividir la respuesta en oraciones
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', response)
    seen = set()
    filtered_sentences = []

    for sentence in sentences:
        normalized_sentence = sentence.strip().lower()
        if normalized_sentence not in seen:
            seen.add(normalized_sentence)
            filtered_sentences.append(sentence)

    return " ".join(filtered_sentences)

# Función para recuperar el contexto relevante de la memoria
def retrieve_context(prompt):
    relevant_memories = [entry["response"] for entry in memory if prompt.lower() in entry["prompt"].lower()]
    if relevant_memories:
        return " ".join(relevant_memories[-3:])  # Recuperar las últimas 3 respuestas relevantes
    return None

# Función para generar una respuesta utilizando el modelo GPT-2 solo si la información no está en la memoria
def generate_response(prompt):
    # Recuperar contexto relevante de la memoria
    context = retrieve_context(prompt)
    if context:
        return context

    # Si no se encuentra en la memoria, generar una respuesta con el modelo
    input_text = f"{prompt}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            temperature=0.6,
            top_p=0.8,
            top_k=30,
            repetition_penalty=2.0,  # Incrementar la penalización de repetición para evitar respuestas redundantes
            do_sample=True
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    response = remove_redundancy(response)

    return response

# Función para añadir interacción a la memoria y guardar en archivo
def add_to_memory(prompt, response):
    memory.append({"prompt": prompt, "response": response})
    with open(memory_file, "w") as f:
        json.dump(memory, f, indent=4)

if __name__ == "__main__":
    # Probar el modelo de la manera más eficiente posible
    while True:
        prompt = input("Tú: ")
        if prompt.lower() in ["salir", "exit", "quit"]:
            break

        # Generar la respuesta utilizando la memoria o el modelo GPT-2
        response_tony = generate_response(prompt)
        print(f"Tony: {response_tony}")

        # Añadir la interacción al dataset y a la memoria persistente
        add_to_memory(prompt, response_tony)
