import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from speechbrain.pretrained import SpeakerRecognition
import torchaudio
from scipy.io import wavfile
import os

torchaudio.set_audio_backend("soundfile")

# Cargar el modelo preentrenado de SpeechBrain
modelo = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_model")

# Carpeta específica para almacenar archivos
CARPETA_AUDIO = "BBDD"

# Crear carpetas si no existen
os.makedirs(CARPETA_AUDIO, exist_ok=True)

# Función para grabar el audio
def grabar_audio(nombre_archivo):
    duracion = 5
    fs = 16000
    ruta_archivo = os.path.join(CARPETA_AUDIO, nombre_archivo)
    print(f"Grabando {ruta_archivo}...")
    audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Grabación terminada.")
    wav.write(ruta_archivo, fs, audio)
    return ruta_archivo

# Función para verificar si las dos grabaciones pertenecen al mismo hablante
def verificar_identidad(archivo1, archivo2):
    # Construir las rutas completas de los archivos
    ruta_archivo1 = os.path.join(CARPETA_AUDIO, archivo1)
    ruta_archivo2 = os.path.join(CARPETA_AUDIO, archivo2)


    # Cargar los dos archivos de audio
    señal1, fs1 = torchaudio.load(ruta_archivo1)
    señal2, fs2 = torchaudio.load(ruta_archivo2)

    # Verificar si las dos voces coinciden
    score, prediccion = modelo.verify_batch(señal1, señal2)
    
    # Resultado de la verificación
    return prediccion

""" def obtener_embebido(archivo_audio):
    # Cargar el archivo de audio
    señal, fs = torchaudio.load(archivo_audio)
    
    # Obtener el embebido (firma de voz)
    embebido = modelo.encode_batch(señal)
    
    return embebido.squeeze().cpu().numpy()  # Convertir a formato NumPy para almacenar fácilmente """

""" def verificar_identidad_con_plantilla(archivo_audio, archivo_plantilla):
    # Cargar el embebido de la plantilla almacenada
    embebido_guardado = np.load(archivo_plantilla)
    
    # Extraer el embebido del nuevo archivo de audio
    embebido_nuevo = obtener_embebido(archivo_audio)
    
    # Calcular la distancia entre los dos embebidos
    distancia = np.linalg.norm(embebido_guardado - embebido_nuevo)
    print (distancia)
    # Definir un umbral de distancia para considerar que las voces coinciden
    umbral = 300  # Este valor puede ajustarse según las necesidades
    if distancia < umbral:
        return True
    else:
        return False
 """






def registrarse():
    usuario = input("ingrese el nombre de usuario: ")
    usuario =  usuario + ".wav"

    # Grabar la primera muestra de audio (usuario registrado)
    grabar_audio(usuario)


    """ embebido_voz = obtener_embebido("usuario_registrado.wav")
    np.save("plantilla_usuario_registrado.npy", embebido_voz) """

def login():
    usuario = input("ingrese el nombre de usuario: ")
    usuario =  usuario + ".wav"

    # Grabar la segunda muestra de audio (intento de autenticación)
    grabar_audio("usuario_actual.wav")
    """ resultado = verificar_identidad_con_plantilla("usuario_actual.wav", "plantilla_usuario_registrado.npy")
    if resultado:
        print("Acceso otorgado")
    else:
        print("Acceso denegado") """
    
    # Verificar si ambas grabaciones corresponden a la misma persona
    if verificar_identidad(usuario, "usuario_actual.wav"):
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
