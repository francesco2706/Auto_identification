import os
import shutil
import random

# Path of the original dataset
original_dir = r"C:\Users\franc\Desktop\data science\PRIMO ANNO\python\final project\Data\genres_original"
# Path of the splitted dataset
output_dir = r"C:\Users\franc\Desktop\data science\PRIMO ANNO\python\final project\Data\splitted_dataset"
train_dir = os.path.join(output_dir, "train")
test_dir  = os.path.join(output_dir, "test")

# Creating directory
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

random.seed(42)
#Genres list: first, list all items in the original_dir, then filter them to include onlt directories. The genres are then sorted.
genres = sorted([g for g in os.listdir(original_dir) if os.path.isdir(os.path.join(original_dir, g))])

#Iterating and splitting the files
for genre in genres:
    #List all files within the current genre_path and filters them to include only .wav files
    genre_path = os.path.join(original_dir, genre)
    files = [f for f in os.listdir(genre_path) if f.endswith(".wav")]
    
    # Randomly reorders the list of files for the current genre. This ensures that train and test are unbiased samples of data
    random.shuffle(files)
    
    # Split 80% of the files for training and the remaining 20% for the test_files
    n_train = int(0.8 * len(files))
    train_files = files[:n_train]
    test_files  = files[n_train:]
    
    # Creating destination directory
    os.makedirs(os.path.join(train_dir, genre), exist_ok=True)
    os.makedirs(os.path.join(test_dir, genre), exist_ok=True)
    
    # Copying files with library shutil 
    for f in train_files:
        shutil.copy(os.path.join(genre_path, f), os.path.join(train_dir, genre, f))
    for f in test_files:
        shutil.copy(os.path.join(genre_path, f), os.path.join(test_dir, genre, f))
print("Splitted datataset correctly generated")
