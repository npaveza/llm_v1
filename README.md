 Descargar optimizer.pt y carpeta Tony desde el siguiente link:
 https://drive.google.com/drive/folders/1AWhkcc5dhAVWRPhEWOlZxcgnVF3BNoG1?usp=drive_link
 Favor de colocar desde carpeta llm para que esté funcional al 100%.
 Razón de esta acción: archivos con peso por sobre el permitido por GITHUB.

 Debe quedar de la siguiente manera:
 llm/Tony/model.safetensors
 llm/Tony/checkpoint-17/model.safetensors
 llm/Tony/checkpoint-17/optimizer.pt

 Funcionalidad:
 - TTS sobre la respuesta del LLM.
 - Micrófono para dictado del PROMPT.
 - Simulación de mecanografía de la respuesta.

 No funciona:
 - Multi-tab de conversaciones, así como abrir nuevos chats o eliminar.
 - Preloader.

 SUGERENCIA:
 Ejecutar con CHROME, al realizar pruebas en FIREFOX lee el mensaje derespuesta EN INGLÉS, a diferenciade CHROME que SI lo lee EN ESPAÑOL.

 El método de aprendizaje es bajo la metodología de Aprendizaje Supervisado, el cual tiene prealmacenada respuestas a ciertas preguntas incluidas en los prompts, no obstante, también puede realizar operaciones lógicas, como multiplicaciones que no están en  los datasets/memory o que sean, por ejemplo, cálculos matemáticos. El motor utilizado es GPT-4 para la generación de respuestas. Las respuestas no anidadas en los archivos json son también guardados como parte del aprendizaje propio para tener respuestas   más rápidas o generar conexión de respuestas para complementar y reforzarlas.
