import openai

# Configurar API key de OpenAI
openai.api_key = 'sk-proj-MGRLbtygqPXT2roYySMIT3BlbkFJALAGbbagZ6UYNaXm2ziO'

# Definir el rol y el contexto inicial para la conversación
DEFAULT_ROLE = "Eres un asistente útil y amigable. Ayudas a los usuarios proporcionando respuestas claras y precisas."
DEFAULT_CONTEXT = "La conversación trata sobre temas de tecnología y desarrollo de software."

def chat_with_gpt4(user_message):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"{DEFAULT_ROLE}\n{DEFAULT_CONTEXT}"},
                {"role": "user", "content": user_message}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        assistant_reply = response['choices'][0]['message']['content']
        return assistant_reply
    except Exception as e:
        return f"Error generando la respuesta: {str(e)}"

if __name__ == "__main__":
    while True:
        user_input = input("Tú: ")
        if user_input.lower() in ["salir", "exit", "quit"]:
            break
        response = chat_with_gpt4(user_input)
        print(f"Asistente: {response}")
