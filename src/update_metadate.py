import pandas as pd

# Chemin vers le fichier des métadonnées
metadata_file = 'H:/Hanyang/Speech_Recognition/PZ/data/annotations/validated.tsv'

# Charger les métadonnées
metadata = pd.read_csv(metadata_file, delimiter='\t')

# Mettre à jour les métadonnées pour pointer vers les fichiers WAV
# Remplacer 'clips/' par 'processed_audio/' et '.mp3' par '.wav'
metadata['path'] = metadata['path'].str.replace('clips/', 'processed_audio/').str.replace('.mp3', '.wav')

# Sauvegarder les métadonnées mises à jour
metadata.to_csv('H:/Hanyang/Speech_Recognition/PZ/data/annotations/metadata_updated.csv', index=False)

# Afficher les premières lignes des métadonnées mises à jour
print("Métadonnées mises à jour :")
print(metadata.head())
