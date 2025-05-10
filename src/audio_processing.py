import os
import librosa
import soundfile as sf

# Dossier contenant les fichiers audio MP3
mp3_dir = 'H:/Hanyang/Speech_Recognition/PZ/data/raw_audio/'

# Dossier pour sauvegarder les fichiers WAV
wav_dir = 'H:/Hanyang/Speech_Recognition/PZ/data/processed_audio/'

# Créer le dossier WAV s'il n'existe pas
os.makedirs(wav_dir, exist_ok=True)

# Convertir les fichiers MP3 en WAV
for file_name in os.listdir(mp3_dir):
    if file_name.endswith('.mp3'):
        mp3_path = os.path.join(mp3_dir, file_name)
        wav_path = os.path.join(wav_dir, file_name.replace('.mp3', '.wav'))

        # Charger l'audio avec Librosa
        audio, sample_rate = librosa.load(mp3_path, sr=None)

        # Sauvegarder l'audio en WAV avec SoundFile
        sf.write(wav_path, audio, sample_rate)
        print(f"Converti : {mp3_path} -> {wav_path}")
