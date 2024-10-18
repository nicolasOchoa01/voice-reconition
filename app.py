import sounddevice as sd
import scipy.io.wavfile as wav
from speechbrain.inference import SpeakerRecognition
import torchaudio
import os
import random
import time 
import openai

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
   

sujetos = [
    "El gato", "La casa", "Un coche", "Una persona", "El perro", "El pájaro", "El niño", "La niña",
    "El profesor", "El estudiante", "El árbol", "La montaña", "El río", "El mar", "El viento",
    "El soldado", "El piloto", "El doctor", "El robot", "El astronauta", "La madre", "El padre",
    "El hermano", "La hermana", "El músico", "El pintor", "El actor", "El escritor", "El ingeniero",
    "El chef", "El guardia", "El policía", "El bombero", "El dragón", "El león", "La mariposa",
    "El tiburón", "El lobo", "El oso", "El ciervo", "El ratón", "El científico", "El investigador",
    "El vecino", "El abogado", "El periodista", "El carpintero", "El mecánico", "El electricista",
    "El granjero", "El pescador", "El cazador", "El ciclista", "El corredor", "El nadador", "El pintor",
    "El escritor", "El explorador", "El aventurero", "El viajero", "El turista", "El guía", "El mago",
    "El rey", "La reina", "El príncipe", "La princesa", "El fantasma", "El vampiro", "El monstruo",
    "El alienígena", "El robot", "El samurái", "El ninja", "El pirata", "El guerrero", "El caballero",
    "El capitán", "El director", "El entrenador", "El jugador", "El bailarín", "El cantante",
    "El músico", "El artista", "El fotógrafo", "El cineasta", "El agricultor", "El tendero",
    "El comerciante", "El banquero", "El administrador", "El político", "El filósofo", "El matemático",
    "El físico", "El químico", "El biólogo", "El arquitecto", "El diseñador", "El programador",
    "El desarrollador", "El técnico", "El operador", "El conductor"
]

# Lista de 100 verbos
verbos = [
    "come", "corre", "salta", "mira", "duerme", "camina", "canta", "baila", "conduce", "nada",
    "vuela", "habla", "escribe", "lee", "juega", "construye", "destruye", "crea", "dibuja", "pinta",
    "explora", "descubre", "investiga", "compra", "vende", "prepara", "lava", "seca", "friega",
    "arregla", "rompe", "abre", "cierra", "enciende", "apaga", "atrapa", "lucha", "cocina", "hornea",
    "traduce", "calcula", "enseña", "aprende", "explica", "colorea", "graba", "escucha", "ve", "observa",
    "detecta", "analiza", "programa", "prueba", "mejora", "crece", "disminuye", "calcula", "esculpe",
    "compone", "interpreta", "crea", "borra", "mueve", "gira", "cae", "se levanta", "gana", "pierde",
    "celebra", "descansa", "invita", "recibe", "viaja", "explora", "descubre", "coloca", "empaqueta",
    "envuelve", "abre", "desempaca", "examina", "analiza", "distribuye", "clasifica", "escoge",
    "selecciona", "envía", "recoge", "saca", "mete", "suelta", "agarra", "espera", "reúne", "separa",
    "conecta", "desconecta", "arma", "desarma", "organiza", "resuelve", "atrapa", "escapa", "protege"
]

# Lista de 100 complementos
complementos = [
    "rápidamente", "en el parque", "bajo la lluvia", "con cuidado", "en silencio", "sin hacer ruido",
    "con alegría", "en la biblioteca", "en la montaña", "en la ciudad", "en el desierto", "en el campo",
    "en el mar", "en el bosque", "en el río", "bajo el sol", "en la sombra", "en el avión", "en el tren",
    "en el coche", "en la bicicleta", "en la playa", "en el estadio", "en el teatro", "en el museo",
    "en la galería", "en la tienda", "en la escuela", "en la universidad", "en el hospital", "en el laboratorio",
    "en la oficina", "en la fábrica", "en el mercado", "en el restaurante", "en el café", "en el bar",
    "en la plaza", "en el jardín", "en la piscina", "en la cancha", "en el gimnasio", "en el zoológico",
    "en la farmacia", "en la estación", "en el aeropuerto", "en la base", "en el puerto", "en el barco",
    "en el submarino", "en la nave espacial", "en la casa", "en el apartamento", "en el edificio",
    "en el rascacielos", "en la cueva", "en la mina", "en la torre", "en el castillo", "en la fortaleza",
    "en la iglesia", "en la catedral", "en el templo", "en la mezquita", "en el mercado", "en el supermercado",
    "en la tienda de ropa", "en la librería", "en la ferretería", "en el banco", "en la peluquería",
    "en la estación de tren", "en la parada de autobús", "en la autopista", "en el túnel", "en el puente",
    "en el parque de atracciones", "en el circo", "en el concierto", "en la conferencia", "en el festival",
    "en la exposición", "en el evento", "en la feria", "en el mercado de pulgas", "en la tienda de comestibles",
    "en la oficina de correos", "en el centro comercial", "en el salón de belleza", "en el spa", "en el club",
    "en la discoteca", "en la heladería", "en la pastelería", "en la panadería", "en la carnicería",
    "en la pescadería", "en la floristería", "en la joyería", "en la tienda de electrónica", "en la gasolinera"
]

def generar_oracion():
    sujeto = random.choice(sujetos)
    verbo = random.choice(verbos)
    complemento = random.choice(complementos)
    texto = f"{sujeto} {verbo} {complemento}"
    print(texto)
    return texto

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
