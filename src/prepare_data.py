import pandas as pd

# Chemin vers le fichier des métadonnées mises à jour
metadata_file = 'H:/Hanyang/Speech_Recognition/PZ/data/annotations/metadata_updated.csv'

# Charger les métadonnées
metadata = pd.read_csv(metadata_file)

# Supprimer les enregistrements avec des labels manquants
metadata = metadata.dropna(subset=['age', 'gender', 'accents'])

# Sélectionner les colonnes nécessaires pour l'entraînement
train_data = metadata[['path', 'sentence', 'age', 'gender', 'accents']]

# Renommer les colonnes pour correspondre à votre format
train_data.columns = ['file_name', 'transcription', 'age', 'gender', 'accent']

# Sauvegarder le fichier CSV pour l'entraînement
train_data.to_csv('H:/Hanyang/Speech_Recognition/PZ/data/annotations/train_data.csv', index=False)

# Afficher les premières lignes du fichier CSV pour l'entraînement
print("Données d'entraînement :")
print(train_data.head())
