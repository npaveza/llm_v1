// Wait for DOM to be fully loaded before accessing elements
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements with null checks
    const chatsContainer = document.getElementById('chats-container');
    const respuestaContainer = document.getElementById('respuesta-container');
    const micButton = document.getElementById('mic-button');
    const preloader = document.getElementById('preloader');
    const userInputElement = document.getElementById('user_input');
    const form = document.querySelector('form');

    // Only show preloader if it exists
    if (preloader) {
        preloader.style.display = 'block';
        setTimeout(() => {
            preloader.style.display = 'none';
        }, 2000);
    }

    // Function to create a new chat
    function createChat(chatId, chatMessage) {
        if (!chatsContainer) return;

        const chatElement = document.createElement('div');
        chatElement.classList.add('chat');
        chatElement.innerHTML = `
            <p>${chatMessage}</p>
            <button class="delete-chat-button">Eliminar</button>
        `;
        chatsContainer.appendChild(chatElement);

        // Add delete chat event
        const deleteChatButton = chatElement.querySelector('.delete-chat-button');
        deleteChatButton.addEventListener('click', () => {
            chatElement.remove();
        });
    }

    // Function to simulate typing response
    function simulateTyping(response) {
        if (!respuestaContainer) return;

        const typingInterval = 50;
        let responseIndex = 0;
        respuestaContainer.innerHTML = ''; // Clear previous response

        const typingFunction = () => {
            if (responseIndex < response.length) {
                const currentLetter = response[responseIndex];
                respuestaContainer.innerHTML += currentLetter;
                responseIndex++;
                setTimeout(typingFunction, typingInterval);
            }
        };

        typingFunction();
    }

    // Function to speak response
    function speakResponse(response) {
        if ('speechSynthesis' in window) {
            const speechSynthesisUtterance = new SpeechSynthesisUtterance(response);
            speechSynthesisUtterance.lang = 'es-ES';
            speechSynthesis.speak(speechSynthesisUtterance);
        }
    }

    // Function to recognize voice
    function recognizeVoice() {
        if (!('webkitSpeechRecognition' in window)) {
            alert('Lo siento, tu navegador no soporta el reconocimiento de voz.');
            return;
        }

        const recognition = new webkitSpeechRecognition();
        recognition.lang = 'es-ES';
        recognition.maxResults = 10;

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            if (userInputElement) {
                userInputElement.value = transcript;
            }
        };

        recognition.onend = () => {
            if (userInputElement) {
                const pregunta = userInputElement.value;
                sendRequest(pregunta);
            }
        };

        recognition.start();
    }

    async function sendRequest(pregunta) {
        try {
            const response = await fetch("/", {
                method: "POST",
                body: new URLSearchParams({ user_input: pregunta }),
                headers: { "Content-Type": "application/x-www-form-urlencoded" }
            });
            
            const data = await response.json();
            const respuesta = data.respuesta;
            
            if (respuestaContainer) {
                respuestaContainer.innerHTML = '';
                simulateTyping(respuesta);
                speakResponse(respuesta);
            }
            
            // Agregar respuesta al contenedor de chat correspondiente
            //const chatId = document.querySelector('.chat-tab.active').id;
            //const chatContent = document.getElementById(`chat-content-${chatId}`);
            //const chatMessage = document.createElement('p');
            //chatMessage.textContent = respuesta;
            //chatContent.appendChild(chatMessage);
        } catch (error) {
            console.error('Error al enviar la solicitud:', error);
        }
    }

    // Add mic button click event
    if (micButton) {
        micButton.addEventListener('click', (event) => {
            event.preventDefault();
            recognizeVoice();
        });
    }

    // Add form submit event
    if (form) {
        form.addEventListener('submit', (event) => {
            event.preventDefault();
            if (userInputElement) {
                const pregunta = userInputElement.value;
                sendRequest(pregunta);
            }
        });
    }

// Function to create chat tabs
function createChatTabs() {
    if (!chatsContainer) return;

    const chatTabsContainer = document.createElement('div');
    chatTabsContainer.classList.add('chat-tabs-container');
    chatTabsContainer.style.position = 'fixed';
    chatTabsContainer.style.top = '100px';
    chatTabsContainer.style.left = '20px';
    chatTabsContainer.style.width = '200px';
    chatTabsContainer.style.background = '#fff';
    chatTabsContainer.style.border = '1px solid #ccc';
    chatTabsContainer.style.borderRadius = '10px';
    chatTabsContainer.style.padding = '20px';
    chatTabsContainer.style.display = 'flex';
    chatTabsContainer.style.flexDirection = 'column';
    chatsContainer.parentNode.appendChild(chatTabsContainer);

    const chatTabs = [];
    for (let i = 0; i < 5; i++) {
        const chatTab = document.createElement('button');
        chatTab.classList.add('chat-tab');
        chatTab.textContent = `Chat ${i + 1}`;
        chatTabsContainer.appendChild(chatTab);
        chatTabs.push(chatTab);

        const chatContent = document.createElement('div');
        chatContent.classList.add('chat-content');
        chatContent.id = `chat-content-${i + 1}`;
        chatContent.style.display = 'none';
        chatsContainer.appendChild(chatContent);

        // Agregar contenido de ejemplo a cada chat
        //const chatMessage = `Este es el contenido del chat ${i + 1}`;
        const chatElement = document.createElement('p');
        chatElement.textContent = chatMessage;
        chatContent.appendChild(chatElement);

        chatTab.addEventListener('click', () => {
            const chats = document.querySelectorAll('.chat-content');
            chats.forEach((chat) => {
                chat.style.display = 'none';
            });
            chatContent.style.display = 'block';
            
            // Agregar clase active al botón de chat seleccionado
            const activeTab = document.querySelector('.chat-tab.active');
            if (activeTab) {
                activeTab.classList.remove('active');
            }
            chatTab.classList.add('active');
        });
    }
}

// Create chat tabs
createChatTabs();
});