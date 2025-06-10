import os
from pydub import AudioSegment

def convert_m4a_to_mp3(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Dossier de sortie créé: {output_folder}")

    for filename in os.listdir(input_folder):
        print(f"Fichier trouvé: {filename}")
        if filename.endswith(".m4a"):
            m4a_file_path = os.path.join(input_folder, filename)
            mp3_filename = os.path.splitext(filename)[0] + ".mp3"
            mp3_file_path = os.path.join(output_folder, mp3_filename)

            print(f"Conversion de: {filename} -> {mp3_filename}")

            try:
                audio = AudioSegment.from_file(m4a_file_path, format="m4a")
                audio.export(mp3_file_path, format="mp3")
                print(f"Converti avec succès: {filename} -> {mp3_filename}")
            except Exception as e:
                print(f"Erreur lors de la conversion de {filename}: {e}")

input_folder = "H:/Hanyang/Speech_Final_Version/m4a_to_mp3/m4a_2" #change the name of the folder if you want to convert some files
output_folder = "H:/Hanyang/Speech_Final_Version/m4a_to_mp3/mp3_2" 
convert_m4a_to_mp3(input_folder, output_folder)
