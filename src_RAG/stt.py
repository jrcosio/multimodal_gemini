import speech_recognition as sr

class EasySpeechRecognizer:
    def __init__(self, energy_threshold=300, pause_threshold=0.8, dynamic_energy_threshold=True):
        """
        Inicializa el reconocedor con parámetros configurables.
        
        Parámetros:
            energy_threshold: Nivel mínimo de energía para considerar el audio como voz.
            pause_threshold: Tiempo de silencio (en segundos) que determina el fin de la intervención.
            dynamic_energy_threshold: Si True, ajusta dinámicamente el umbral de energía.
        """
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.dynamic_energy_threshold = dynamic_energy_threshold
        
    def calibrate(self, duration=1):
        """
        Calibra el reconocedor para el ruido ambiental.
        
        Parámetros:
            duration: Duración en segundos para realizar la calibración.
        """
        with sr.Microphone() as source:
            print("Calibrando el ruido ambiental...")
            self.recognizer.adjust_for_ambient_noise(source, duration=duration)
            print(f"Calibración completada: energy_threshold={self.recognizer.energy_threshold}")
            
    def listen_and_recognize(self, language="es-ES"):
        """
        Escucha a través del micrófono y convierte la voz a texto utilizando el servicio de Google.
        
        Parámetros:
            language: Código de idioma (por defecto español: "es-ES").
            
        Retorna:
            El texto reconocido o None en caso de error.
        """
        with sr.Microphone() as source:
            print("Habla ahora...")
            audio = self.recognizer.listen(source)
        try:
            texto = self.recognizer.recognize_google(audio, language=language)
            print("Texto reconocido:", texto)
            return texto
        except sr.UnknownValueError:
            print("No se pudo entender el audio.")
            return None
        except sr.RequestError as e:
            print(f"Error al conectarse al servicio: {e}")
            return None

# Ejemplo de uso
if __name__ == "__main__":
    recognizer = EasySpeechRecognizer(energy_threshold=300, pause_threshold=1.0)
    recognizer.calibrate(duration=1)
    texto = recognizer.listen_and_recognize(language="es-ES")
