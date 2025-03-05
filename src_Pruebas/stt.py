import speech_recognition as sr

# Inicializa el reconocedor
r = sr.Recognizer()

# Configura el umbral de energía (nivel mínimo para considerar que es voz)
r.energy_threshold = 300  # Puedes ajustar este valor según el ambiente

# Configura el tiempo de silencio antes de finalizar la escucha
r.pause_threshold = 0.8  # Por defecto es 0.8 segundos

# Utiliza el micrófono como fuente de audio
with sr.Microphone() as source:
    print("Habla ahora...")
    audio = r.listen(source)

# Intenta reconocer el texto usando el servicio de Google
try:
    # Especifica el idioma, en este caso español
    texto = r.recognize_google(audio, language="es-ES")
    print("Has dicho:", texto)
except sr.UnknownValueError:
    print("No se pudo entender el audio")
except sr.RequestError as e:
    print("Error al conectarse al servicio de reconocimiento; {0}".format(e))

