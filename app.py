import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from speechbrain.pretrained import SpeakerRecognition
import torchaudio
from scipy.io import wavfile

torchaudio.set_audio_backend("soundfile")

# Función para grabar el audio
def grabar_audio(archivo):
    duracion = 5
    fs = 16000
    print(f"Grabando {archivo}...")
    audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Grabación terminada.")
    wav.write(archivo, fs, audio)

# Función para verificar si las dos grabaciones pertenecen al mismo hablante
def verificar_identidad(archivo1, archivo2):
    # Cargar el modelo preentrenado de SpeechBrain
    modelo = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_model")

    # Cargar los dos archivos de audio
    señal1, fs1 = torchaudio.load(archivo1)
    señal2, fs2 = torchaudio.load(archivo2)

    # Verificar si las dos voces coinciden
    score, prediccion = modelo.verify_batch(señal1, señal2)
    
    # Resultado de la verificación
    return prediccion



# Grabar la primera muestra de audio (usuario registrado)
grabar_audio("usuario_registrado.wav")

# Grabar la segunda muestra de audio (intento de autenticación)
grabar_audio("usuario_actual.wav")

# Verificar si ambas grabaciones corresponden a la misma persona
if verificar_identidad("usuario_registrado.wav", "usuario_actual.wav"):
    print("Acceso otorgado")
else:
    print("Acceso denegado")
