import os
import librosa
import soundfile as sf

# Paths to directories
RAW_AUDIO_DIR = 'data/raw_audio'
PROCESSED_AUDIO_DIR = 'data/processed_audio'

# Preprocessing parameters
SAMPLE_RATE = 16000  # Sample rate

def normalize_audio(audio):
    """Normalize the audio to have an amplitude between -1 and 1."""
    return librosa.util.normalize(audio)

def preprocess_audio(file_path, output_dir):
    """Load, normalize, and save an audio file."""
    try:
        # Load the audio
        audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
        print(f"Loaded audio file: {file_path}")  # Checkpoint

        # Normalize the audio
        audio = normalize_audio(audio)
        print(f"Normalized audio: {file_path}")  # Checkpoint

        # Save the processed audio
        base_name = os.path.basename(file_path).replace('.wav', '')
        output_path = os.path.join(output_dir, f"{base_name}_processed.wav")
        sf.write(output_path, audio, sample_rate)
        print(f"Saved processed audio: {output_path}")  # Checkpoint

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def preprocess_all_audio(input_dir, output_dir):
    """Preprocess all audio files in a directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(input_dir, file_name)
            preprocess_audio(file_path, output_dir)

# Run the preprocessing
preprocess_all_audio(RAW_AUDIO_DIR, PROCESSED_AUDIO_DIR)
print("Preprocessing complete.")  # Checkpoint
