import sounddevice as sd
import scipy.io.wavfile as wav
from speechbrain.inference import SpeakerRecognition
import torchaudio
import os
import random
#import time 
import openai
from listas import sujetos, verbos, complementos

# cargar la api key desde varible de entorno
openai.api_key = os.getenv('openai.api_key')


# configuracion de torchaudio para usar un backend de audio compatible con wav
torchaudio.set_audio_backend("soundfile")

# cargar el modelo preentrenado de SpeechBrain
modelo = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_model", use_auth_token=False)


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
    return ruta_archivo

def generar_oracion():
    sujeto = random.choice(sujetos)
    verbo = random.choice(verbos)
    complemento = random.choice(complementos)
    texto = f"{sujeto} {verbo} {complemento}"
    print(texto)
    return texto

def transcribir(ruta):
    # Enviar el archivo a Whisper para la transcripción
    with open(ruta, 'rb') as audio:
        transcripcion  = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio
        )
    # Obtener el texto transcrito
    transcripcion_text = transcripcion.text
    # print("Transcripción:", transcripcion_text)
    return transcripcion_text


def validar_lectura(transcripcion, texto):
    if transcripcion == texto:
        return True
    else:
        return False

def validar_voz(archivo1, archivo2):
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
    
# función para verificar si las dos grabaciones pertenecen al mismo hablante
def verificar_identidad(transcripcion, texto, archivo1, archivo2):
    
    if validar_lectura(transcripcion, texto):
        print("lectura correcta")
        if validar_voz(archivo1, archivo2):
            print("voz correcta")
            return True
        else:
            print("voz incorrecta")
            return False
    else:
        print("lectura incorrecta")
        return False


def registrarse():
    usuario = input("ingrese el nombre de usuario: ")
    grabar_audio(usuario)

def login():
    # primero pide el nombre de usuario para buscarlo en la base de datos
    usuario = input("ingrese el nombre de usuario: ")
    # genera la oracion a leer y abre el microfono para leerla
    texto = generar_oracion()
    ruta_audio = grabar_audio("usuario_actual")
    # transcribe el audio de lectura
    transcripcion = transcribir(ruta_audio)

    # realiza la validacion tanto de la lectura como de la voz
    if verificar_identidad(transcripcion, texto, usuario, "usuario_actual"):
        print("Acceso otorgado")
    else:
        print("Acceso denegado")

while True:
    print("Menú Principal:")
    print("1) registrarse")
    print("2) login")
    print("3) Salir")
    opcion = input("Selecciona una opción: ")

    if opcion == "1":
        registrarse()
    elif opcion == "2":
        login()
    elif opcion == "3":
        print("Saliendo del programa...")
        break
    else:
        print("Opción no válida. Inténtalo de nuevo.")
