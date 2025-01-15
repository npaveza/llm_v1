import torch
from sentence_transformers import SentenceTransformer, util
import json
import os
import random

class MemoryModule:
    """Clase que almacena y gestiona la memoria de las interacciones anteriores."""

    def __init__(self, memory_file="tony_memory.json", dataset_file="tony_dataset.json"):
        self.memory_file = memory_file
        self.dataset_file = dataset_file
        self.memory = self.load_memory()
        self.dataset = self.load_dataset()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Modelo para codificación semántica

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
        """Recuperar contexto relevante de la memoria y el dataset utilizando similitud semántica."""
        current_embedding = self.embedding_model.encode(current_prompt, convert_to_tensor=True)

        # Similitud con el archivo de memoria
        similarities_memory = [
            (entry, util.pytorch_cos_sim(current_embedding, self.embedding_model.encode(entry["prompt"], convert_to_tensor=True)).item())
            for entry in self.memory
        ]

        # Similitud con el dataset
        similarities_dataset = [
            (entry, util.pytorch_cos_sim(current_embedding, self.embedding_model.encode(entry["prompt"], convert_to_tensor=True)).item())
            for entry in self.dataset
        ]

        # Combinar ambas similitudes y ordenar por relevancia
        combined_similarities = similarities_memory + similarities_dataset
        combined_similarities = sorted(combined_similarities, key=lambda x: x[1], reverse=True)

        # Obtener las respuestas de los tres prompts más similares
        relevant_memories = [entry[0] for entry in combined_similarities[:3]]
        context = " ".join([entry["response"] for entry in relevant_memories])
        return context


class TonyNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TonyNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu2 = torch.nn.ReLU()
        self.attention = torch.nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)

        attn_output, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x = attn_output.squeeze(0)

        x = self.fc3(x)
        return self.softmax(x)


class TonyTester:
    def __init__(self, model_path='tony_nn.pth', memory_file='tony_memory.json', dataset_file='tony_dataset.json'):
        self.memory_module = MemoryModule(memory_file=memory_file, dataset_file=dataset_file)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.recent_responses = []  # Historial de respuestas recientes

        input_size = 384
        hidden_size = 64
        output_size = input_size

        self.model = TonyNN(input_size, hidden_size, output_size)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        else:
            self.train_neural_network(model_path)

    def get_vector_representation(self, prompt):
        """Obtener representación vectorial del prompt usando el modelo de embeddings."""
        embedding = self.embedding_model.encode(prompt, convert_to_tensor=True)
        return embedding

    def diversify_response(self, response):
        """Modificar ligeramente la respuesta para hacerla más diversa."""
        variations = [
            "Espero haberte ayudado. Aquí tienes tu respuesta: ",
            "Esta es mi respuesta: ",
            "Permíteme asistirte: ",
            "Te puedo ayudar con esto: "
        ]
        return f"{random.choice(variations)}{response}"

    def retrieve_context(self, prompt):
        """Recuperar contexto relevante basado en similitud de embeddings."""
        return self.memory_module.retrieve_context(prompt)

    def ask_tony(self, prompt):
        # Obtener el embedding del prompt
        prompt_embedding = self.get_vector_representation(prompt)

        # Recuperar contexto relevante basado en similitud semántica
        context = self.retrieve_context(prompt)

        if context:
            # Evitar respuestas repetidas
            if context in self.recent_responses:
                print(f"Contexto relevante encontrado, pero ya usado recientemente: {context}")
            else:
                print(f"Contexto relevante encontrado: {context}")
                self.recent_responses.append(context)
                if len(self.recent_responses) > 3:  # Limitar a las últimas 3 respuestas
                    self.recent_responses.pop(0)
        else:
            print("No se encontró contexto relevante en la memoria ni en el dataset.")

        # Alimentar el embedding del prompt a la red neuronal de Tony
        with torch.no_grad():
            prompt_embedding = prompt_embedding.unsqueeze(0)  # Añadir dimensión batch
            tony_response_embedding = self.model(prompt_embedding)

        # Recuperar el embedding del contexto relevante
        context_embedding = self.get_vector_representation(context)

        # Comparar la similitud entre el embedding generado y el contexto
        similarity = util.pytorch_cos_sim(tony_response_embedding, context_embedding).item()

        # Ajustar el umbral de similitud a 0.5 para ser más permisivo
        if similarity > 0.5:
            return self.diversify_response(context)  # Diversificar la respuesta para mayor variedad
        else:
            return "Tony: Lo siento, no encontré una respuesta adecuada."

    def train_neural_network(self, model_path):
        """Entrenar la red neuronal de Tony si no se encuentra el archivo de modelo."""
        print("[INFO] Entrenando la red neuronal de Tony...")

        # Preparar datos de entrenamiento a partir del dataset
        input_texts = [entry["prompt"] for entry in self.memory_module.dataset]
        target_texts = [entry["response"] for entry in self.memory_module.dataset]

        inputs = [self.get_vector_representation(text).unsqueeze(0) for text in input_texts]
        targets = [self.get_vector_representation(text).unsqueeze(0) for text in target_texts]

        inputs = torch.cat(inputs, dim=0)
        targets = torch.cat(targets, dim=0)

        # Configurar entrenamiento
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        epochs = 10  # Ajusta este valor según tus necesidades
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            print(f"[INFO] Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

        # Guardar el modelo entrenado
        torch.save(self.model.state_dict(), model_path)
        print(f"[INFO] Modelo TonyNN entrenado y guardado en {model_path}")


# Inicializar TonyTester y realizar preguntas
if __name__ == "__main__":
    tester = TonyTester()

    while True:
        user_input = input("Tú: ")
        if user_input.lower() in ["salir", "exit"]:
            break

        response = tester.ask_tony(user_input)
        print(response)
