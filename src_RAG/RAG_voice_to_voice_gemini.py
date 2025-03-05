import asyncio
import os
import sys
import traceback
import pyaudio
import argparse

from google import genai
from dotenv import load_dotenv
from rag import Rag
from stt import EasySpeechRecognizer

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup
    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup
    
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024


MODEL = "models/gemini-2.0-flash-exp"

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY2")

client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})

TIPO = "AUDIO"          #"TEXTO" o "AUDIO"

SYSTEM_MESSAGE = "Eres un asistente inteligente. Ayudas a los usuarios siempre en Español y siempre dentro del contexto exclusivo de los libros de aventuras. Si no encuentras información relevante, responde honestamente."

#CONFIG = {"generation_config": {"response_modalities": ["TEXT"]}}

CONFIG = {
    "system_instruction": {
        "parts": [{"text": SYSTEM_MESSAGE}]
    },
    "generation_config": {
        "response_modalities": ["AUDIO"],
        "speech_config": "Aoede"   #Aoede, Charon, Fenrir, Kore y Puck.
    }
}

pya = pyaudio.PyAudio()

class AudioLoop:
    def __init__(self):
        self.audio_in_queue = None
        self.session = None

    async def listen_voice_command(self):
        """
        Utiliza EasySpeechRecognizer para capturar el audio del usuario, convierte la voz a texto,
        obtiene fragmentos relevantes mediante rag y envía el mensaje a la API.
        """
        # Instancia y calibración del reconocedor
        recognizer = EasySpeechRecognizer(energy_threshold=300, pause_threshold=1.0, dynamic_energy_threshold=True)
        await asyncio.to_thread(recognizer.calibrate, duration=1)
        while True:
            text = await asyncio.to_thread(recognizer.listen_and_recognize, language="es-ES")
            if text is None:
                continue
            if text.lower().strip() == "salir":
                print("Saliendo... pulsa Ctrl+C")
                break

            # Obtener fragmentos relevantes (contexto) usando rag
            context_chunk = rag.get_chunk_relevates(text)
            context = "\n".join(context_chunk)

            # Construir el prompt con contexto
            mensaje_con_contexto = f"""Eres un asistente inteligente diseñado exclusivamente para apoyar lectores de libros de aventuras, 
                para el analisis y hacer resumenes de los libros proporcionados. Ayudas a los usuarios siempre en Español 
                y siempre dentro del contexto exclusivo de los libros de aventuras. Si no encuentras información relevante, responde honestamente.
                
                ### **Pautas Clase:**
                **Responde eclusivamente en ESPAÑOL.** No respondas en otro idioma.
                **Responde siempre dentro del contexto exclusivo de los libros de aventuras.**
                **Si no encuentras información relevante, responde honestamente.**
                **Si no entiendes la pregunta, responde honestamente.**
                **Si no puedes responder, responde honestamente.**
                **Cuando respondas, intenta ser lo más claro y conciso posible.**

                Contexto:
                {context}

                Pregunta: {text}

                Respuesta:"""
            # Enviar el mensaje a la sesión
            await self.session.send(input=mensaje_con_contexto or ".", end_of_turn=True)

    async def receive_audio(self):
        """
        Recibe la respuesta de la API y, si contiene audio, lo pone en cola para reproducirlo;
        en caso de texto, lo muestra en consola.
        """
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")
            # Limpia la cola de audio si se ha interrumpido la respuesta.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        """
        Reproduce el audio recibido a través de un stream de PyAudio.
        """
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()

                # Se crea una tarea para el reconocimiento de voz y envío del mensaje
                tg.create_task(self.listen_voice_command())
                # Se crean tareas para recibir y reproducir la respuesta (audio)
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                # Espera hasta que listen_voice_command finalice (p.ej., al escribir "q")
                await asyncio.Event().wait()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            traceback.print_exception(e)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Se han eliminado los argumentos de modo (visión)
    args = parser.parse_args()
    
    rag = Rag('./libros/harrypotter_caliz.txt')
    
    main = AudioLoop()
    asyncio.run(main.run())
