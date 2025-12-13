import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models 
from tqdm import tqdm
from torchmetrics import Accuracy, Precision, Recall 
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import random
# --- CONFIGURATION ---
NUM_CLASSES = 10 
BATCH_SIZE = 60
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
IMG_SIZE = 128           
VAL_SPLIT_RATIO = 0.20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- RANDOM MANAGEMENT ---
def set_seed(seed): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- PATHS ---
# Update these paths if moving the script
train_data_dir = r'C:\Users\giann\Desktop\universita\magistrale\FUNDATIONS OF DATA SCIENCE\progetto finale\Data\Dataset_Spectrogram\train'
test_data_dir = r'C:\Users\giann\Desktop\universita\magistrale\FUNDATIONS OF DATA SCIENCE\progetto finale\Data\Dataset_Spectrogram\test'


# --- DATA TRANSFORMS ---
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- RESNET18 MODEL SETUP ---
def get_resnet_model(num_classes, device):
    print("Downloading and configuring ResNet18...")
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    # Modify the final Fully Connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return model.to(device)

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
    set_seed(42)
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
    model = get_resnet_model(NUM_CLASSES, DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. TRAINING LOOP
    print("\n--- START TRAINING RESNET18 ---")
    
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
