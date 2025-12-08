import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from torchmetrics import Accuracy, Precision, Recall 
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURATION ---
NUM_CLASSES = 10 
BATCH_SIZE = 60
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
IMG_SIZE = 128           
VAL_SPLIT_RATIO = 0.20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- PATHS ---
train_data_dir = r'C:\Users\giann\Desktop\universita\magistrale\FUNDATIONS OF DATA SCIENCE\progetto finale\Data\Dataset_Spectrogram\train'
test_data_dir = r'C:\Users\giann\Desktop\universita\magistrale\FUNDATIONS OF DATA SCIENCE\progetto finale\Data\Dataset_Spectrogram\test'

# --- DATA TRANSFORMS ---
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- MODELLO CNN SETUP ---
class EfficientSpectrogramCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(EfficientSpectrogramCNN, self).__init__()

        # Blocco 1: 128x128 -> 64x64
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        # Blocco 2: 64x64 -> 32x32
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calcolo dimensione per il livello lineare
        # Input: 32 canali * 32 larghezza * 32 altezza
        self.fc_input_size = 32 * 32 * 32 
        
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x) 
        x = self.block2(x) 
        x = self.flatten(x)
        x = self.classifier(x) 
        return x

# --- VALIDATION FUNCTION ---
def validate_epoch(model, dataloader, device):
    model.eval()
    acc_metric = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            acc_metric.update(preds, targets)
    return acc_metric.compute().item()

# --- TEST FUNCTION ---
def test_model(model, dataloader, device, class_names):
    acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)
    precision = Precision(task="multiclass", num_classes=NUM_CLASSES, average='macro').to(device)
    recall = Recall(task="multiclass", num_classes=NUM_CLASSES, average='macro').to(device)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            acc.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print(f"\nTest Accuracy: {acc.compute():.4f}")
    print(f"Test Macro Precision: {precision.compute():.4f}")
    print("\nClassification Report (Test Set):") 
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8)) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.show()
    
    return acc.compute()


# ====================================================================
# --- MAIN EXECUTION LOOP ---
# ====================================================================

if __name__ == '__main__':    
    # 1. LOAD DATA
    full_train_dataset = datasets.ImageFolder(root=train_data_dir, transform=data_transforms)
    test_dataset = datasets.ImageFolder(root=test_data_dir, transform=data_transforms)
    
    val_size = int(VAL_SPLIT_RATIO * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, validation_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    class_names = full_train_dataset.classes
    print(f"Dataset Training: {len(train_dataset)} | Validation: {len(validation_dataset)} | Test: {len(test_dataset)}")

    # 2. INITIALIZATION
    model = EfficientSpectrogramCNN(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_acc = 0.0

    # 3. TRAINING LOOP
    print("\n--- INIZIO TRAINING EFFICIENTE ---")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for data, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            
            # Forward  
            scores = model(data)
            loss = criterion(scores, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * data.size(0)

        total_train_loss = epoch_loss / len(train_dataset)
        val_acc = validate_epoch(model, validation_loader, DEVICE)
        
        print(f"Loss: {total_train_loss:.4f} | Val Acc: {val_acc:.4f}")
        
       
    # 4. FINAL EVALUATION
    test_model(model, test_loader, DEVICE, class_names)
