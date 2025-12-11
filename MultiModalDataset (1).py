#Questo file gestisce la lettura deid ati da due directory separate (spec_root e wave_root)

from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torch
import os

# --- CLASSE CUSTOM DATASET ---
class MultiModalDataset(Dataset):
    def __init__(self, spec_root, wave_root, spec_transform=None, wave_transform=None):
        
        # Carica le liste dei percorsi da entrambe le directory usando ImageFolder
        # Assumiamo che le due directory abbiano le stesse sottocartelle (classi) e gli stessi file
        self.spec_dataset = datasets.ImageFolder(spec_root)
        self.wave_dataset = datasets.ImageFolder(wave_root)
        
        # Le liste dei percorsi e delle etichette (samples) devono corrispondere
        self.spec_paths = self.spec_dataset.samples 
        self.wave_paths = self.wave_dataset.samples 
        
        self.spec_transform = spec_transform
        self.wave_transform = wave_transform
        self.class_names = self.spec_dataset.classes
        
        # Controllo di coerenza
        if len(self.spec_paths) != len(self.wave_paths):
             print(f"ERRORE: Spectrogrammi trovati: {len(self.spec_paths)}, Waveform trovate: {len(self.wave_paths)}")
             raise AssertionError("I dataset devono avere lo stesso numero di elementi e ordine (file corrispondenti)!")
        
    def __len__(self):
        return len(self.spec_paths)

    #per un dato indice idx carica due immagini spectro e wave dai rispettivi percorsi 
    #e l'etichetta (label). Queste sono convertite in RGB per essere compatibili con i pesi pre-addestrati 
    # di ResNet18 
    def __getitem__(self, idx):
        # 1. Caricamento file e etichetta (Label)
        spec_path, label = self.spec_paths[idx]
        wave_path, _ = self.wave_paths[idx]
        
        # 2. Caricamento Immagini (assicurandosi che siano RGB)
        # Il .convert('RGB') è una sicurezza nel caso in cui PIL le legga come L (grayscale)
        spec_img = Image.open(spec_path).convert('RGB')
        wave_img = Image.open(wave_path).convert('RGB') 
        
        # 3. Trasformazioni
        if self.spec_transform:
            spec_img = self.spec_transform(spec_img)
        if self.wave_transform:
            wave_img = self.wave_transform(wave_img)
            
        # Restituisce i due input e l'unica etichetta
        return spec_img, wave_img, label

# crea gli oggetti MultiModalDataset per Train e Test. Esegue lo split del set di training
# in set di training e validation utilizzando random_split e avvolge tutti e tre i set negli oggetti DataLoader per 
# l'iterazione in batch durante l'addestramento
def get_dataloaders(spec_train_dir, wave_train_dir, spec_test_dir, wave_test_dir, 
                    spec_transforms, wave_transforms, batch_size, val_split_ratio=0.20):
    
    # Dataset di Training e Validazione (set completo da splittare)
    full_train_dataset = MultiModalDataset(
        spec_root=spec_train_dir, wave_root=wave_train_dir,
        spec_transform=spec_transforms, wave_transform=wave_transforms
    )

    # Dataset di Test
    test_dataset = MultiModalDataset(
        spec_root=spec_test_dir, wave_root=wave_test_dir,
        spec_transform=spec_transforms, wave_transform=wave_transforms
    )

    # Split Train/Validation
    val_size = int(val_split_ratio * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, validation_dataset = random_split(full_train_dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    class_names = full_train_dataset.class_names
    
    return train_loader, validation_loader, test_loader, class_names