import streamlit as st
import os
import librosa
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jamo import h2j, j2hcj
import re
from jamotools import join_jamos
import numpy as np
from pydub import AudioSegment

# Function to convert m4a to wav
def convert_m4a_to_wav(file_path):
    audio = AudioSegment.from_file(file_path, format="m4a")
    wav_path = file_path.replace(".m4a", ".wav")
    audio.export(wav_path, format="wav")
    return wav_path

# Define the model
class MultiTaskWav2Vec2(nn.Module):
    def __init__(self, base_model_name, num_age, num_gender, num_accent):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(base_model_name)
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
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)

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

# Convert jamo to Korean text
def text_to_jamo(text):
    return ''.join(j2hcj(h2j(char)) for char in text)

def clean_jamo_text(jamo_text):
    return re.sub(r'[^\u3131-\u3163\u1100-\u11FF\uAC00-\uD7A3 ]', '', jamo_text)

def jamo_to_korean(jamo_text):
    return join_jamos(jamo_text)  # Converting Jamo to Hangul (Korean characters)

def main():
    st.set_page_config(layout="wide")
    
    # Title with some spacing
    st.markdown("<h1 style='text-align:center; color: #003366;'>PZ: Accent Prediction App</h1>", unsafe_allow_html=True)

    # Load the model
    model_path = 'H:/Hanyang/Speech_Final_Version/model/best_model_augmentation.pt'
    base_model_name = "student-47/wav2vec2-large-xlrs-korean-v5"
    num_age = 3  # twenties, thirties, fourties
    num_gender = 2  # male, female
    num_accent = 6  # France, Singapore, China, Switzerland, South Korea, Ukrainian

    model = load_model(model_path, base_model_name, num_age, num_gender, num_accent)

    # Dictionaries to convert numerical predictions to text labels
    age_labels = {0: 'twenties', 1: 'thirties', 2: 'fourties'}
    gender_labels = {0: 'female', 1: 'male'}
    accent_labels = {
        0: 'France',
        1: 'Singapore',
        2: 'China',
        3: 'Switzerland',
        4: 'South Korea',
        5: 'Ukrainian'
    }

    # Layout: Split into two columns
    col1, col2 = st.columns([1, 2])

    with col1:
        # Upload file section with a stylish box around it
        st.markdown("""
            <div style="border: 2px solid #003366; padding: 15px; border-radius: 8px;">
                <h5 style="color: #003366; font-weight: bold;">Upload an Audio File (WAV, MP3, M4A)</h5>
                <p style="color: #666;">Choose an audio file to start the prediction process.</p>
            </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])
        if uploaded_file is not None:
            # Save the uploaded file
            file_path = os.path.join('uploads', uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            st.success('File uploaded successfully')

            # If the file is m4a, convert it to wav
            if file_path.endswith('.m4a'):
                file_path = convert_m4a_to_wav(file_path)

    with col2:
        # Prediction button with interactive hover effect
        if st.button('Make Prediction'):
            if uploaded_file is not None:
                # Preprocess the audio
                audio, sr = preprocess_audio(file_path)
                input_values = model.processor(audio, return_tensors="pt", sampling_rate=sr).input_values

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

                    # Get the text transcription
                    logits = outputs['logits']
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = model.processor.batch_decode(predicted_ids)[0]

                    # Convert jamo to Korean text
                    korean_text = text_to_jamo(transcription)
                    korean_text = clean_jamo_text(korean_text)
                    korean_text = jamo_to_korean(korean_text)  # Convert Jamo to Hangul (Korean)

                st.success('Prediction completed successfully')

                # Displaying the results in a neat card style
                st.markdown(f"""
                    <div style="background-color: #f4f4f9; padding: 20px; border-radius: 10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);">
                        <h3 style="font-size: 18px; font-weight: bold; color: #003366;">Model Prediction</h3>
                        <p><strong>Text:</strong> {korean_text}</p>
                        <p><strong>Age:</strong> {age_labels[age_pred]} (Confidence: {age_confidence:.2f})</p>
                        <p><strong>Gender:</strong> {gender_labels[gender_pred]} (Confidence: {gender_confidence:.2f})</p>
                        <p><strong>Accent:</strong> {accent_labels[accent_pred]} (Confidence: {accent_confidence:.2f})</p>
                    </div>
                """, unsafe_allow_html=True)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    main()
