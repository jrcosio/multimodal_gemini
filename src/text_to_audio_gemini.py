import asyncio
import base64
import contextlib
import datetime
import os
import json
import wave
import itertools
from dotenv import load_dotenv
from google import genai
import shutil

from IPython.display import display, Audio

MODEL = "gemini-2.0-flash-exp"

# https://ai.google.dev/gemini-api/docs/multimodal-live?hl=es-419
# https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI.ipynb

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})

def borrar_todos_los_audios(folder_path):
    """ 
        Elimina todos los archivos y directorios en la carpeta especificada.
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


@contextlib.contextmanager #Abre y cierra el archivo de audio (Recurso)
def wave_file(filename, channels=1, rate=24000, sample_width=2):
    """
        Context manager para crear y configurar un archivo de audio WAV.

        Args:
            filename (str): El nombre del archivo WAV a crear.
            channels (int, opcional): El número de canales de audio. Por defecto es 1 (mono).
            rate (int, opcional): La tasa de muestreo del audio en Hz. Por defecto es 24000 Hz.
            sample_width (int, opcional): El ancho de muestra en bytes. Por defecto es 2 bytes.

        Yields:
            wave.Wave_write: Un objeto wave.Wave_write configurado para escribir datos de audio en el archivo especificado.
    """
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf

async def async_enumerate(it):
    """
        Enumera asincrónicamente un iterable asincrónico.
        Args:
            it (AsyncIterable): Un iterable asincrónico a enumerar.
        Yields:
            Tuple[int, Any]: Una tupla que contiene el índice actual y el elemento correspondiente del iterable.
        Ejemplo:
            async for index, item in async_enumerate(some_async_iterable):
                print(index, item)
    """
    n = 0
    async for item in it:
        yield n, item
        n +=1

SYSTEM_MESSAGE = "Eres un asistente experto en tecnología y ayudas a los usuarios siempre en Español."
  
config = {
    "system_instruction": {
        "parts": [{"text": SYSTEM_MESSAGE}]
    },
    "generation_config": {
        "response_modalities": ["AUDIO"],
        "speech_config": "Puck"   #Aoede, Charon, Fenrir, Kore y Puck.
    }
}

async def main():
    borrar_todos_los_audios('./audios')
    numero = 0
    async with client.aio.live.connect(model=MODEL, config=config) as session:
        
        while True:
            message = input("\nHablame--> ")
            if message.lower() == "exit":
                break
            numero += 1
            file_name = f'./audios/audio_{numero}.wav'
            
            with wave_file(file_name) as wav:    
                
                await session.send(input=message, end_of_turn=True)

                turn = session.receive()
                async for n,response in async_enumerate(turn):
                    if response.data is not None:
                        wav.writeframes(response.data)

                        if n==0:
                            print(response.server_content.model_turn.parts[0].inline_data.mime_type)
                        print('.', end='')


    #display(Audio(file_name, autoplay=True)) # Esto es para mostrar el audio en el notebook
    
if __name__ == "__main__":
    asyncio.run(main())