import asyncio
import os
import sys
import traceback

import pyaudio
import argparse

from google import genai
from dotenv import load_dotenv
from rag import Rag

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

SYSTEM_MESSAGE = "Eres un asistente experto en tecnología y ayudas a los usuarios siempre en Español."

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
        self.out_queue = None
        self.session = None
        self.send_text_task = None
        
    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "Mensaje > ",
            )
            if text.lower() == "q":
                break
            
             # Obtener fragmentos relevantes
            context_chunk = rag.get_chunk_relevates(text)
            context = "\n".join(context_chunk)
            
            # Construir el prompt que se enviará a la API
            mensaje_con_contexto = f"""Eres un asistente de IA que responde basándote en el contexto proporcionado. Si no encuentras información relevante, responde honestamente.

                Contexto:
                {context}

                Pregunta: {text}

                Respuesta:
            """
            
            
            await self.session.send(input=mensaje_con_contexto or ".", end_of_turn=True)
        
    
    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        # En modo debug se desactiva la excepción por overflow.
        kwargs = {"exception_on_overflow": False} if __debug__ else {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
    
    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)
    
    async def receive_audio(self):
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
                self.out_queue = asyncio.Queue(maxsize=5)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                # Se mantiene la ejecución hasta que se cancele la tarea.
                #await asyncio.Event().wait()
                await send_text_task
                raise asyncio.CancelledError("User requested exit")
                
            

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.audio_stream.close()
            traceback.print_exception(e)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Se han eliminado los argumentos de modo (visión)
    args = parser.parse_args()
    
    rag = Rag('./libros/harrypotter_caliz.txt')
    
    main = AudioLoop()
    asyncio.run(main.run())
