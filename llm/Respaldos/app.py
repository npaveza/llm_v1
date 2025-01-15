import openai
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Configurar API key de OpenAI
openai.api_key = ''

class TonyNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TonyNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  # Añadir dropout para prevenir el sobreajuste
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return self.softmax(x)

class InteractiveTrainerWithReasoning:
    """Clase que permite interactuar con el modelo y entrenarlo dinámicamente, promoviendo el razonamiento."""

    def __init__(self, model_name='gpt2', output_dir='./Tony', learning_rate=2e-5, batch_size=4, num_epochs=1, weight_decay=0.01):
        self.model_name = model_name
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)

        # Añadir un token de relleno (pad token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))  # Ajustar el tamaño de las embeddings si se añade un nuevo token

        # Cargar dataset si existe
        self.dataset = self.load_dataset()

    def load_dataset(self):
        """Cargar el dataset desde un archivo JSON si existe."""
        if os.path.exists("tony_dataset.json"):
            with open("tony_dataset.json", "r") as f:
                return json.load(f)
        return []

    def save_dataset(self):
        """Guardar el dataset en un archivo JSON."""
        with open("tony_dataset.json", "w") as f:
            json.dump(self.dataset, f, indent=4)

    def add_interaction(self, prompt, response):
        """Añadir una nueva interacción al dataset y entrenar el modelo."""
        # Validar la respuesta antes de añadirla al dataset
        if not response.strip() or response.strip().isdigit():
            print("[ERROR] La respuesta generada no es válida. No se añadirá al dataset.")
            return

        self.dataset.append({"prompt": prompt, "response": response})
        print(f"[INFO] Añadiendo interacción al dataset: \nPrompt: {prompt}\nRespuesta de GPT-4: {response}\n")
        
        # Guardar la interacción en el archivo JSON
        self.save_dataset()

        # Entrenar el modelo de manera incremental
        self.train_incrementally()

        # Entrenar la red neuronal con los datos actuales
        self.train_neural_network()

    def generate_response_with_gpt4(self, prompt):
        """Generar una respuesta utilizando GPT-4 para guiar a Tony."""
        # Utilizar la API de GPT-4 para obtener una respuesta inicial
        role = "Eres un asistente llamado Tony Creado Por Victor Cherif, experto en programación, ciencia, y filosofía. Eres amigable, paciente y siempre dispuesto a ayudar a otros a aprender. Respondes de manera precisa y siempre en español."
        context = "Proporcionas respuestas claras y detalladas sobre programación, desarrollo de software, e incluso debates filosóficos sobre la tecnología. Eres experto en Python, Flask, y siempre respondes en un tono amigable y didáctico."

        messages = [
            {"role": "system", "content": f"{role}\n{context}"},
            {"role": "user", "content": prompt}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )

        assistant_reply = response['choices'][0]['message']['content'].strip()
        return assistant_reply

    def train_incrementally(self):
        """Entrenar el modelo de manera incremental con los datos actuales."""
        if not self.dataset:
            return

        print("[INFO] Iniciando entrenamiento incremental...")

        train_data = [{"input_text": entry["prompt"], "target_text": entry["response"]} for entry in self.dataset]
        tokenized_datasets = Dataset.from_dict({
            "input_text": [entry["input_text"] for entry in train_data],
            "target_text": [entry["target_text"] for entry in train_data]
        })

        def tokenize_function(examples):
            inputs = [input_text + self.tokenizer.eos_token + target_text for input_text, target_text in zip(examples["input_text"], examples["target_text"])]
            model_inputs = self.tokenizer(inputs, truncation=True, padding='max_length', max_length=512)
            # Copiar los input_ids a los labels
            model_inputs['labels'] = model_inputs['input_ids'].copy()
            return model_inputs

        tokenized_datasets = tokenized_datasets.map(tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="no",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=self.weight_decay,
            logging_dir='./logs',
            save_total_limit=1,
            fp16=True,  # Usar FP16 si tu GPU lo permite para reducir el uso de memoria y acelerar el entrenamiento
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets,
        )

        trainer.train()
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)  # Guardar el tokenizador también
        print(f"[INFO] Modelo Tony entrenado y guardado en {self.output_dir}")

    def train_neural_network(self):
        """Entrenar una Red Neuronal Artificial con los datos actuales del dataset."""
        if not self.dataset:
            return

        print("[INFO] Iniciando entrenamiento de la Red Neuronal Artificial...")

        # Preparar datos
        input_texts = [entry["prompt"] for entry in self.dataset]
        target_texts = [entry["response"] for entry in self.dataset]

        # Convertir textos en vectores usando el tokenizer
        inputs = [self.tokenizer.encode(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128) for text in input_texts]
        targets = [self.tokenizer.encode(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128) for text in target_texts]

        inputs = torch.cat(inputs, dim=0)
        targets = torch.cat(targets, dim=0)

        # Convertir los targets a tipo float para que coincidan con los outputs
        targets = targets.float()

        # Configurar parámetros de la RNA
        input_size = inputs.shape[1]
        hidden_size = 64  # Ajusta este valor según tus necesidades
        output_size = input_size  # Salida de igual dimensión que la entrada

        model = TonyNN(input_size, hidden_size, output_size)
        criterion = nn.MSELoss()  # Cambiar a MSELoss
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Entrenar el modelo
        epochs = 10  # Ajusta este valor según tus necesidades
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            outputs = model(inputs.float())
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            print(f"[INFO] Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

        # Guardar el modelo
        torch.save(model.state_dict(), 'tony_nn.pth')
        print("[INFO] Red Neuronal Artificial de Tony entrenada y guardada.")

if __name__ == "__main__":
    # Inicializar el entrenador interactivo con un modelo más liviano
    interactive_trainer = InteractiveTrainerWithReasoning(
        model_name="gpt2",  # Cambiado a "gpt2" para usar un modelo más liviano
        output_dir='./Tony',
        learning_rate=3e-5,
        batch_size=2,
        num_epochs=1,
        weight_decay=0.01
    )

    while True:
        prompt = input("Tú: ")
        if prompt.lower() in ["salir", "exit", "quit"]:
            break

        # Obtener respuesta de GPT-4 para alimentar el modelo de Tony
        response_gpt4 = interactive_trainer.generate_response_with_gpt4(prompt)
        print(f"GPT-4: {response_gpt4}")

        # Añadir la interacción de GPT-4 al dataset y entrenar a Tony
        interactive_trainer.add_interaction(prompt, response_gpt4)

        # Generar respuesta con el modelo entrenado Tony
        response_tony = response_gpt4  # Usar directamente la respuesta de GPT-4 como respuesta de Tony
        print(f"Tony: {response_tony}")
