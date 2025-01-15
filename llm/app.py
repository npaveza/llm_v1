import torch
import torch.nn as nn
import torch.optim as optim
import openai
import json
from sentence_transformers import SentenceTransformer, util
import random
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math

# Configurar API key de OpenAI
openai.api_key = 'sk-dcru9veuokTpZscsvqF54YDe0w4jFPVHnXTAbuBjzTT3BlbkFJXDKNeqw4U6L_QPPQKaAtdFveJfzZVkhIkk_2XWwhUA'

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TonyNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads=4, max_len=512):
        super(TonyNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        # Positional Encoding para capturar el orden de los tokens
        self.positional_encoding = PositionalEncoding(d_model=hidden_size, max_len=max_len)

        # Múltiples capas de atención
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)

        # Dos capas más para capturar patrones complejos
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Añadir el encoding posicional para secuencias
        x = self.positional_encoding(x)
        
        # Aplicar atención multi-cabeza en secuencias
        attn_output, _ = self.attention(x, x, x)

        x = attn_output
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.softmax(x)

class MemoryModule:
    def __init__(self, memory_file="tony_memory.json", dataset_file="tony_dataset.json"):
        self.memory_file = memory_file
        self.dataset_file = dataset_file
        self.memory = self.load_memory()
        self.dataset = self.load_dataset()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                return json.load(f)
        return []

    def load_dataset(self):
        if os.path.exists(self.dataset_file):
            with open(self.dataset_file, "r") as f:
                return json.load(f)
        return []

    def save_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=4)

    def save_dataset(self):
        with open(self.dataset_file, "w") as f:
            json.dump(self.dataset, f, indent=4)

    def add_memory(self, prompt, response):
        self.memory.append({"prompt": prompt, "response": response})
        self.save_memory()

    def add_to_dataset(self, prompt, response):
        self.dataset.append({"prompt": prompt, "response": response})
        self.save_dataset()

    def retrieve_context(self, current_prompt):
        if not current_prompt.strip():
            return ""
            
        current_embedding = self.embedding_model.encode(current_prompt, convert_to_tensor=True)
        
        # Filtrar memorias vacías o inválidas
        valid_memories = [m for m in self.memory if m["prompt"].strip() and m["response"].strip()]
        
        similarities = []
        for entry in valid_memories:
            try:
                entry_embedding = self.embedding_model.encode(entry["prompt"], convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(current_embedding, entry_embedding).item()
                similarities.append((entry, similarity))
            except Exception as e:
                print(f"Error processing entry: {e}")
                continue
                
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        relevant_memories = similarities[:3] if similarities else []
    
        return " ".join([entry[0]["response"] for entry in relevant_memories])

class TonyTesterWithSequences:
    def __init__(self, model_name='DeepESP/gpt2-spanish', output_dir='./Tony', model_path='tony_nn.pth'):
        self.model_name = model_name
        self.output_dir = output_dir
        self.memory_module = MemoryModule()
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))

        input_size = 384
        hidden_size = 64
        output_size = input_size
        self.tony_nn = TonyNN(input_size, hidden_size, output_size)
        if os.path.exists(model_path):
            self.tony_nn.load_state_dict(torch.load(model_path))
            self.tony_nn.eval()
        else:
            self.train_neural_network(model_path)

        self.recent_responses = []

    def normalize_input(self, prompt):
        """Normaliza el input para evitar diferencias menores entre variantes."""
        return prompt.lower().strip()

    def diversify_response(self, response):
        variations = [
            "Aquí está mi respuesta: ",
            "Basado en mi conocimiento: ",
            "Te respondo lo siguiente: ",
            "De acuerdo a mi análisis: ",
            "Mi respuesta es: "
        ]
        
        # Evitar repetir la última variación usada
        if hasattr(self, 'last_variation'):
            variations = [v for v in variations if v != self.last_variation]
        
        selected_variation = random.choice(variations)
        self.last_variation = selected_variation
        
        return f"{selected_variation}{response}"

    def generate_response_with_gpt4(self, prompt, temperature=0.7, max_tokens=150):
        context = self.memory_module.retrieve_context(prompt)
        messages = [
            {"role": "system", "content": "Eres Tony, un asistente de IA creado por Victor Cherif"},
            {"role": "user", "content": f"Contexto previo: {context}\n\nPregunta actual: {prompt}"}
        ]
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Lo siento, hubo un error: {str(e)}"

    def ask_tony(self, prompt, temperature=1.0, max_tokens=150):
        # Normalizar el input
        prompt = self.normalize_input(prompt)
        
        #NUEVO CODIGO PARA PROBAR
        normalized_prompt = self.normalize_input(prompt)
        # Obtener el embedding del prompt actual
        prompt_embedding = self.memory_module.embedding_model.encode(normalized_prompt, convert_to_tensor=True)
        context = self.memory_module.retrieve_context(normalized_prompt)

        # Obtener respuesta de GPT-4 con el prompt real
        response_gpt4 = self.generate_response_with_gpt4(
            prompt=normalized_prompt,  # Usar el prompt normalizado
            temperature=temperature,
            max_tokens=max_tokens
        )
        #FIN NUEVO CODIGO PARA PROBAR

        # Añadir respuesta a la memoria y al dataset
        self.memory_module.add_memory(prompt, response_gpt4)
        self.memory_module.add_to_dataset(prompt, response_gpt4)

        # También entrenar a Tony después de la interacción con GPT-4
        self.train_tony(prompt, response_gpt4)

        return f"GPT-4: {response_gpt4}"

    def train_tony(self, prompt, response):
        input_embedding = self.memory_module.embedding_model.encode(prompt, convert_to_tensor=True).unsqueeze(0)
        response_embedding = self.memory_module.embedding_model.encode(response, convert_to_tensor=True).unsqueeze(0)

        inputs = torch.cat([input_embedding], dim=0)
        targets = torch.cat([response_embedding], dim=0)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.tony_nn.parameters(), lr=0.001)

        self.tony_nn.train()
        optimizer.zero_grad()

        outputs = self.tony_nn(inputs)

        #Ajuste para coincidir outputs:
        outputs = outputs.squeeze(1)
        #Termino ajuste

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        print(f"[INFO] Tony entrenado con interacción. Loss: {loss.item()}")

    def train_neural_network(self, model_path):
        input_texts = [entry["prompt"] for entry in self.memory_module.dataset]
        target_texts = [entry["response"] for entry in self.memory_module.dataset]

        inputs = [self.memory_module.embedding_model.encode(text, convert_to_tensor=True).unsqueeze(0) for text in input_texts]
        targets = [self.memory_module.embedding_model.encode(text, convert_to_tensor=True).unsqueeze(0) for text in target_texts]

        inputs = torch.cat(inputs, dim=0)
        targets = torch.cat(targets, dim=0)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.tony_nn.parameters(), lr=0.001)

        epochs = 10
        for epoch in range(epochs):
            self.tony_nn.train()
            optimizer.zero_grad()

            outputs = self.tony_nn(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            print(f"[INFO] Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

        torch.save(self.tony_nn.state_dict(), model_path)
        print(f"[INFO] Red Neuronal Tony entrenada y guardada en {model_path}")

# Código nuevo para funcionar con FLASK - FrontEnd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

tony_tester = TonyTesterWithSequences()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        pregunta = request.form["user_input"]
        respuesta = tony_tester.ask_tony(prompt=pregunta)
        return jsonify({'respuesta': respuesta})
    return render_template("index.html", respuesta="")

@app.route('/ajax', methods=['POST'])
def handle_ajax():
    data = request.get_json()
    # Procesar la data aquí
    respuesta = 'Respuesta desde el servidor'
    return jsonify({'respuesta': respuesta})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
# Fin código nuevo para funcionar con FLASK - FrondEnd