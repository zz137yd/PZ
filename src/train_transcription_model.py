import os
import torch
import pandas as pd
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Configuration pour utiliser le GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du device: {device}")

# Chemin vers les fichiers audio et les annotations
audio_dir = 'H:/Hanyang/Speech_Recognition/PZ/data/processed_audio/'
annotations_file = 'H:/Hanyang/Speech_Recognition/PZ/data/annotations/train_data.csv'

# Charger les annotations
annotations = pd.read_csv(annotations_file)

# Charger le modèle et le processeur Whisper
model_name = "openai/whisper-large"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

# Fonction pour transcrire un fichier audio
def transcribe_audio(file_path):
    # Charger l'audio
    audio, sample_rate = librosa.load(file_path, sr=None)

    # Prétraiter l'audio et déplacer sur GPU
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt").to(device)

    # Générer la transcription
    with torch.no_grad():
        output = model.generate(inputs.input_values)

    # Décoder la transcription
    transcription = processor.batch_decode(output, skip_special_tokens=True)
    return transcription[0]

# Transcrire tous les fichiers audio
for index, row in annotations.iterrows():
    file_path = os.path.join(audio_dir, row['file_name'])
    transcription = transcribe_audio(file_path)
    print(f"Transcription pour {file_path}: {transcription}")
