import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from speechbrain.pretrained import SpeakerRecognition
import torchaudio
import os

# configuracion de torchaudio para usar soundfile
torchaudio.set_audio_backend("soundfile")

# cargar el modelo preentrenado de SpeechBrain
modelo = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_model")

# carpeta que simula una base de datos para los audios
CARPETA_AUDIO = "BBDD"

# crea la carpeta BBDD si no existe
os.makedirs(CARPETA_AUDIO, exist_ok=True)

# función para grabar un audio de 5 segundos
def grabar_audio(nombre_archivo):
    duracion = 5
    fs = 16000
    ruta_archivo = os.path.join(CARPETA_AUDIO, nombre_archivo + ".wav")
    print(f"Grabando {ruta_archivo}...")
    audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Grabación terminada.")
    wav.write(ruta_archivo, fs, audio)

# función para verificar si las dos grabaciones pertenecen al mismo hablante
def verificar_identidad(archivo1, archivo2):
    # construccion de las rutas de ambos archivos
    ruta_archivo1 = os.path.join(CARPETA_AUDIO, archivo1 + ".wav")
    ruta_archivo2 = os.path.join(CARPETA_AUDIO, archivo2 + ".wav")

    # cargar los dos archivos de audio
    señal1, fs1 = torchaudio.load(ruta_archivo1)
    señal2, fs2 = torchaudio.load(ruta_archivo2)

    # verificar si las dos voces coinciden
    score, prediccion = modelo.verify_batch(señal1, señal2)
    
    # resultado de la verificación, valor booleano
    return prediccion

def registrarse():
    usuario = input("ingrese el nombre de usuario: ")
    grabar_audio(usuario)

def login():
    usuario = input("ingrese el nombre de usuario: ")
    grabar_audio("usuario_actual")
    
    # Verificar si ambas grabaciones corresponden a la misma persona
    if verificar_identidad(usuario, "usuario_actual"):
        print("Acceso otorgado")
    else:
        print("Acceso denegado")

while True:
    print("Menú Principal:")
    print("1. registrarse")
    print("2. login")
    print("3. Salir")
    opcion = input("Selecciona una opción (1-3): ")

    if opcion == "1":
        registrarse()
    elif opcion == "2":
        login()
    elif opcion == "3":
        print("Saliendo del programa...")
        break
    else:
        print("Opción no válida. Inténtalo de nuevo.")
