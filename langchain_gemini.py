from dotenv import load_dotenv
import os,asyncio
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp") 


async def main():
    while True:
        message = input("\nHablame--> ")
        if message.lower() == "exit":
            break
        
        messages = [
            ("system", "Eres un asistente que ayuda a los usuarios en Español."),
            ("human", message)
        ]
        response = llm.invoke(messages)
        print(f"Gemini: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())