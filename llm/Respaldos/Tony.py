from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn as nn
import json
import os

class TonyNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TonyNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

class TonyConversationalModel:
    """Clase para interactuar con el modelo Tony ya entrenado."""

    def __init__(self, model_path='./Tony', dataset_path='tony_dataset.json', nn_model_path='./MyModels/tony_nn.pth'):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.nn_model_path = '/home/victhor/Dev/MyModels/tony_nn.pth'
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_path)

        # Asegurarse de que el tokenizer tenga un pad token configurado
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Inicializar el historial de la conversación
        self.conversation_history = []

        # Cargar el dataset entrenado
        self.dataset = self.load_dataset()

        # Cargar el modelo de Red Neuronal Artificial
        self.nn_model = self.load_neural_network_model()

    def load_dataset(self):
        """Cargar el dataset desde un archivo JSON si existe."""
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, "r") as f:
                return json.load(f)
        return []

    def load_neural_network_model(self):
        """Cargar el modelo de Red Neuronal Artificial entrenado."""
        input_size = 128  # Debe coincidir con el tamaño de entrada utilizado al entrenar el modelo TonyNN
        hidden_size = 64  # Debe coincidir con el tamaño oculto utilizado al entrenar TonyNN
        output_size = input_size  # La salida tiene el mismo tamaño que la entrada

        nn_model = TonyNN(input_size, hidden_size, output_size)
        nn_model.load_state_dict(torch.load(self.nn_model_path, weights_only=True))
        nn_model.eval()
        return nn_model

    def find_similar_prompt(self, prompt):
        """Buscar en el dataset si hay un prompt similar al que se proporcionó."""
        for entry in self.dataset:
            if prompt.lower() in entry["prompt"].lower():
                return entry["response"]
        return None

    def enhance_with_neural_network(self, input_text):
        """Utilizar el modelo de Red Neuronal para mejorar el contexto de la respuesta."""
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        with torch.no_grad():
            output = self.nn_model(input_ids.float())
        enhanced_text = self.tokenizer.decode(torch.argmax(output, dim=-1)[0])
        return enhanced_text

    def generate_response_with_model(self, prompt):
        """Generar una respuesta usando el modelo Tony con el historial de la conversación."""
        # Buscar en el dataset una respuesta similar
        similar_response = self.find_similar_prompt(prompt)
        if similar_response:
            print("[INFO] Respuesta encontrada en el dataset.")
            self.conversation_history.append(f"Tú: {prompt}")
            self.conversation_history.append(f"Tony: {similar_response}")
            return similar_response

        # Agregar el mensaje del usuario al historial de la conversación
        self.conversation_history.append(f"Tú: {prompt}")

        # Formar el historial de la conversación como contexto
        context = "\n".join(self.conversation_history) + "\nTony: "

        # Tokenizar el historial de la conversación completo
        input_ids = self.tokenizer.encode(context, return_tensors='pt')

        # Limitar la longitud máxima del historial para no superar la capacidad del modelo
        if len(input_ids[0]) > 768:  # Puedes ajustar el límite según el tamaño del modelo
            input_ids = input_ids[:, -768:]  # Solo mantén los últimos 768 tokens

        # Generar la respuesta utilizando el modelo GPT-2
        output = self.model.generate(
            input_ids,
            max_length=len(input_ids[0]) + 50,
            num_return_sequences=1,
            temperature=0.5,  # Reducir la aleatoriedad para mejorar consistencia
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extraer solo la parte de la respuesta de Tony
        response = response[len(context):].strip()

        # Mejorar la respuesta utilizando la Red Neuronal Artificial
        enhanced_response = self.enhance_with_neural_network(response)

        # Validar que la respuesta no esté vacía o sea repetitiva
        if not enhanced_response or enhanced_response.lower() in ["the the", ""]:
            enhanced_response = "Lo siento, podrías intentar reformular la pregunta."

        # Agregar la respuesta al historial
        self.conversation_history.append(f"Tony: {enhanced_response}")
        return enhanced_response

if __name__ == "__main__":
    tony_model = TonyConversationalModel(model_path='./Tony')

    while True:
        prompt = input("Tú: ")
        if prompt.lower() in ["salir", "exit", "quit"]:
            break

        # Generar la respuesta con el modelo entrenado Tony
        response = tony_model.generate_response_with_model(prompt)
        print(f"Tony: {response}")
