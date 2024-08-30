import librosa
import soundfile as sf
import numpy as np
import os

carpeta_salida = "C:/Users/MM/OneDrive/Documentos/Alzheimer/Data audio/Audios_splited"
if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)

def particion_audio(audio_path):
    audio, sr = librosa.load(audio_path)
    audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Definir la duración del segmento
    duracion_segmento = 10 * sr  # 10 segundos en muestras, sr muestras en un segundo
    cota = 5 * sr                  #5 segundos en muestras
    # Calcular el número de muestras necesarias para completar el último segmento
    muestras_sobrantes = len(audio) % duracion_segmento
    num_muestras_faltantes = duracion_segmento - (muestras_sobrantes)
    # Verificar si el número de muestras faltantes es menor que 5 segundos de muestras
    if muestras_sobrantes < cota:
        # Eliminar el último segmento si es menor que 5 segundos de muestras
        audio = audio[:-(len(audio) % duracion_segmento)]
    else:
        # Agregar muestras de silencio al final del audio si es necesario
        if num_muestras_faltantes != duracion_segmento:
            audio = np.concatenate((audio, np.zeros(num_muestras_faltantes)))
    
    a = 0   #contador
    # Dividir el audio en segmentos y guardarlos en la carpeta de salida
    for i in range(0, len(audio), duracion_segmento):
        segmento = audio[i:i+duracion_segmento]
        nombre_archivo = os.path.join(carpeta_salida, f"{audio_filename}_Parte {a}.wav")
        sf.write(nombre_archivo, segmento, sr)
        a=a+1

audio_dir="C:/Users/MM/OneDrive/Documentos/Alzheimer/Data audio/audios/"
for audio_filaname in os.listdir(audio_dir):
    audio_path =audio_dir + audio_filaname
    particion_audio(audio_path)
