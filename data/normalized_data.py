import os
import librosa
import soundfile as sf
from pydub import AudioSegment

# Paths to folders
raw_folder = 'H:/Hanyang/Speech_Final_Version/data/raw'  # Folder containing raw audio files
processed_folder = 'H:/Hanyang/Speech_Final_Version/data/processed'  # Folder where normalized files will be saved

# Create the 'processed' folder if it doesn't exist
if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)

# Function to convert .m4a files to .wav
def convert_m4a_to_wav(m4a_path, wav_path):
    audio = AudioSegment.from_file(m4a_path, format="m4a")
    audio.export(wav_path, format="wav")

# Iterate through all files in the 'raw' folder
for filename in os.listdir(raw_folder):
    if filename.endswith('.mp3') or filename.endswith('.m4a'):
        # Paths for input and output files
        input_filepath = os.path.join(raw_folder, filename)
        output_filename = os.path.splitext(filename)[0] + '.wav'
        output_filepath = os.path.join(processed_folder, output_filename)

        # Convert .m4a files to .wav
        if filename.endswith('.m4a'):
            convert_m4a_to_wav(input_filepath, output_filepath)
        else:
            # Load the audio file
            audio, sr = librosa.load(input_filepath, sr=None)

            # Normalize the audio
            audio_normalized = librosa.util.normalize(audio)

            # Save the normalized file in .wav format
            sf.write(output_filepath, audio_normalized, sr)

print("Conversion and normalization of audio files completed.")
