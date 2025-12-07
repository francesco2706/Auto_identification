import os
import shutil
import random

# Percorsi
original_dir = r"C:\Users\franc\Desktop\data science\PRIMO ANNO\python\final project\Data\genres_original"
output_dir = r"C:\Users\franc\Desktop\data science\PRIMO ANNO\python\final project\Data\splitted_dataset"

train_dir = os.path.join(output_dir, "train")
test_dir  = os.path.join(output_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Set random seed per riproducibilità
random.seed(42)

# Lista dei generi
genres = sorted([g for g in os.listdir(original_dir) if os.path.isdir(os.path.join(original_dir, g))])

for genre in genres:
    genre_path = os.path.join(original_dir, genre)
    files = [f for f in os.listdir(genre_path) if f.endswith(".wav")]
    
    # Shuffle random
    random.shuffle(files)
    
    # Split 80/20
    n_train = int(0.8 * len(files))
    train_files = files[:n_train]
    test_files  = files[n_train:]
    
    # Crea cartelle train/test per il genere
    os.makedirs(os.path.join(train_dir, genre), exist_ok=True)
    os.makedirs(os.path.join(test_dir, genre), exist_ok=True)
    
    # Copia i file
    for f in train_files:
        shutil.copy(os.path.join(genre_path, f), os.path.join(train_dir, genre, f))
    for f in test_files:
        shutil.copy(os.path.join(genre_path, f), os.path.join(test_dir, genre, f))

print("Dataset splittato correttamente!")
