import streamlit as st
import os
import librosa
import soundfile as sf
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, AutoProcessor
import numpy as np

# Define the model
class MultiTaskWav2Vec2(nn.Module):
    def __init__(self, base_model_name, num_age, num_gender, num_accent):
        super().__init__()
        self.asr = Wav2Vec2ForCTC.from_pretrained(base_model_name)
        hidden_size = self.asr.config.hidden_size

        # Classification heads
        self.age_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, num_age)
        )
        self.gender_head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, num_gender)
        )
        self.accent_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, num_accent)
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.asr.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, T, H]
        pooled = hidden_states.mean(dim=1)  # [B, H]

        age_logits = self.age_head(pooled)
        gender_logits = self.gender_head(pooled)
        accent_logits = self.accent_head(pooled)
        ctc_logits = self.asr.lm_head(hidden_states)

        return {
            'logits': ctc_logits,
            'age_logits': age_logits,
            'gender_logits': gender_logits,
            'accent_logits': accent_logits,
        }

# Load the model
def load_model(model_path, base_model_name, num_age, num_gender, num_accent):
    model = MultiTaskWav2Vec2(base_model_name, num_age, num_gender, num_accent)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess audio
def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    audio = librosa.util.normalize(audio)
    return audio, sr

def main():
    st.title('Upload a File')

    # Load the model
    model_path = 'H:/Hanyang/Speech_Recognition/PZ/models/best_xlsr-model.pt'
    base_model_name = "student-47/wav2vec2-large-xlrs-korean-v5"
    num_age = 4  # Update with the actual number of age classes used during training
    num_gender = 2  # Replace with the actual number of gender classes
    num_accent = 8  # Update with the actual number of accent classes used during training

    model = load_model(model_path, base_model_name, num_age, num_gender, num_accent)
    processor = AutoProcessor.from_pretrained(base_model_name)

    # Example dictionaries to convert numerical predictions to text labels
    age_labels = {0: 'twenties', 1: 'thirties', 2: 'forties', 3: 'fifties'}
    gender_labels = {0: 'female_feminine', 1: 'male_masculine'}
    accent_labels = {0: 'Seoul', 1: 'Busan', 2: 'Daegu', 3: 'Incheon', 4: 'Gwangju', 5: 'Daejeon', 6: 'Ulsan', 7: 'Sejong'}

    # Upload a file of any type
    uploaded_file = st.file_uploader("Choose a file", type=None)
    if uploaded_file is not None:
        # Save the uploaded file
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        st.success('File uploaded successfully')
        st.write(f"File name: {uploaded_file.name}")
        st.write(f"File type: {uploaded_file.type}")
        st.write(f"File size: {uploaded_file.size} bytes")

        # Button to make the prediction
        if st.button('Make Prediction'):
            # Convert the file to WAV
            audio, sr = preprocess_audio(file_path)
            wav_path = file_path.replace(os.path.splitext(file_path)[1], '.wav')
            sf.write(wav_path, audio, sr)

            # Preprocess the audio
            input_values = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

            # Make a prediction with the model
            with torch.no_grad():
                outputs = model(input_values=input_values)
                age_logits = outputs['age_logits']
                gender_logits = outputs['gender_logits']
                accent_logits = outputs['accent_logits']

                age_pred = torch.argmax(age_logits, dim=1).item()
                gender_pred = torch.argmax(gender_logits, dim=1).item()
                accent_pred = torch.argmax(accent_logits, dim=1).item()

                # Calculate confidence scores
                age_confidence = torch.softmax(age_logits, dim=1)[0][age_pred].item()
                gender_confidence = torch.softmax(gender_logits, dim=1)[0][gender_pred].item()
                accent_confidence = torch.softmax(accent_logits, dim=1)[0][accent_pred].item()

                # Get the text transcription in hangul
                logits = outputs['logits']
                predicted_ids = torch.argmax(logits, dim=-1)
                predicted_text = processor.batch_decode(predicted_ids)[0]

            st.success('Prediction completed successfully')
            st.write("=== Model Prediction ===")
            st.write(f"Text   : {predicted_text}")
            st.write(f"Age    : {age_labels[age_pred]} (Confidence: {age_confidence:.2f})")
            st.write(f"Gender : {gender_labels[gender_pred]} (Confidence: {gender_confidence:.2f})")
            st.write(f"Accent : {accent_labels[accent_pred]} (Confidence: {accent_confidence:.2f})")

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    main()
