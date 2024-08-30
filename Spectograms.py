import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import random

def spec_augment(spec: np.ndarray, num_mask=1, 
                 freq_masking_max_percentage=0.30, time_masking_max_percentage=0.30):

    spec = spec.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)  #porcentage de ancho de banda eliminar
        
        num_freqs_to_mask = int(freq_percentage * all_freqs_num)  #tamaño del ancho de banda a eliminar
        f0 = random.uniform(0.0, all_freqs_num - num_freqs_to_mask) #punto aleatorio sobre el eje vertical
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = 0   #Cero toda la banda horizontal

        time_percentage = random.uniform(0.0, time_masking_max_percentage)
        
        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = random.uniform(0.0, all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = 0
    
    return spec

def audio_segment_to_mel_spectrogram_rgb(audio_path, output_dir, img_width, img_height, label=False):
    # Load the audio file
    y, sr = librosa.load(audio_path)
    
    # Get the original audio file name without extension 
    audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Generate mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y.astype(float), sr=sr, S=None, n_fft=1024, hop_length=128, n_mels=128)

    # Convert to decibels
    mel_spec = librosa.power_to_db(mel_spectrogram, ref=np.max)     #Transforma el espectrograma mel en decibelios
    
    if label:
        Spec = spec_augment(mel_spectrogram)
        spec_db = librosa.power_to_db(Spec, ref=np.max)
        # Plot mel-spectrogram without axes, title, or colorbar
        plt.figure()   
        librosa.display.specshow(spec_db, sr=sr, cmap="hsv")        
        
        # Save the image with the same name as the audio file 
        output_image_path = os.path.join(output_dir, f'{audio_filename}-spec.png')  
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi = 300) 
        plt.close()
        log_S = cv.imread(output_image_path)
        log_S_resized = cv.resize(log_S, (img_width, img_height), interpolation=cv.INTER_CUBIC)
        
        cv.imwrite(output_image_path, log_S_resized)
    
    # Plot mel-spectrogram without axes, title, or colorbar
    plt.figure()   
    librosa.display.specshow(mel_spec, sr=sr, cmap="hsv")       
         
    # Save the image with the same name as the audio file 
    output_image_path = os.path.join(output_dir, f'{audio_filename}.png') 
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=300) 
    plt.close()
    
    # Load the saved image (Cargar la imagen guardada)
    log_S = cv.imread(output_image_path)

    # Resize the image using cv2.resize (Redimensiona la imagen)
    log_S_resized = cv.resize(log_S, (img_width, img_height), interpolation=cv.INTER_CUBIC)
  
    # Save the resized image
    cv.imwrite(output_image_path, log_S_resized)



audio_dir = "C:/Users/MM/OneDrive/Documentos/Alzheimer/Data audio/Audios_splited"
output_dir = "C:/Users/MM/OneDrive/Documentos/Alzheimer/Data audio/Spectograms_hsv"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print("Directorio creado: ", output_dir)

# Specify the dimensions for the resized image (in pixels)
img_width = 256  # Change this value as needed
img_height = 256  # Change this value as needed
files_names = os.listdir(audio_dir)

# Loop through the audio files and convert specified audio segments to mel-spectrogram images
for audio_file in files_names:
    audio_path = os.path.join(audio_dir, audio_file)
    if audio_file.find('AD')==0 or audio_file.find('MCI')==0:  
        audio_segment_to_mel_spectrogram_rgb(audio_path, output_dir, img_width, img_height, label=True)
    else:
        audio_segment_to_mel_spectrogram_rgb(audio_path, output_dir, img_width, img_height)
