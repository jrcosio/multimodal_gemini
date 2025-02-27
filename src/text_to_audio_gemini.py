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

from IPython.display import display, Audio

MODEL = "gemini-2.0-flash-exp"

# https://ai.google.dev/gemini-api/docs/multimodal-live?hl=es-419
# https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI.ipynb

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})


@contextlib.contextmanager
def wave_file(filename, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf

async def async_enumerate(it):
    n = 0
    async for item in it:
        yield n, item
        n +=1
        
config={
    "generation_config": {
        "response_modalities": ["AUDIO"]
        }
    }

async def main():
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


    display(Audio(file_name, autoplay=True))
    
if __name__ == "__main__":
    asyncio.run(main())