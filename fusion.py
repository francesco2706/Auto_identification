import torch
from torch import optim
from torch import nn
from torchvision import transforms, models 
from tqdm import tqdm
from torchmetrics import Accuracy, Precision, Recall 
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import random
from MultiStreamResNet import MultiStreamResNet 
from MultiModalDataset import get_dataloaders 
def set_seed(seed): 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

NUM_CLASSES = 10 
BATCH_SIZE = 32 
LEARNING_RATE = 0.0005 
NUM_EPOCHS = 20
IMG_SIZE = 128
VAL_SPLIT_RATIO = 0.20
MODEL_SAVE_PATH = 'multimodal_resnet_fusion.pth'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

base_dir = r'C:\Users\franc\Desktop\data science\PRIMO ANNO\python\final project'

spec_train_dir = os.path.join(base_dir, r'Dataset_Spectrogram\train')
spec_test_dir = os.path.join(base_dir, r'Dataset_Spectrogram\test')
wave_train_dir = os.path.join(base_dir, r'Dataset_Waveform\train')
wave_test_dir = os.path.join(base_dir, r'Dataset_Waveform\test')

standard_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def validate_epoch(model, dataloader, device):
    model.eval()
    acc_metric = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)
    with torch.no_grad():
        for spec_data, wave_data, targets in dataloader:
            spec_data, wave_data, targets = spec_data.to(device), wave_data.to(device), targets.to(device)
            outputs = model(spec_data, wave_data) 
            _, preds = torch.max(outputs, 1)
            acc_metric.update(preds, targets)
    return acc_metric.compute().item()

def test_model(model, dataloader, device, class_names):
    acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)
    precision = Precision(task="multiclass", num_classes=NUM_CLASSES, average='macro', zero_division=0).to(device)
    recall = Recall(task="multiclass", num_classes=NUM_CLASSES, average='macro', zero_division=0).to(device)
    
    model.eval()
    all_preds = []
    all_labels = []

    print("\n*** Starting final evaluation")
    
    with torch.no_grad():
        for spec_data, wave_data, labels in tqdm(dataloader, desc="Testing Multi-Stream"):
            
            spec_data = spec_data.to(device)
            wave_data = wave_data.to(device)
            labels = labels.to(device)
            
            outputs = model(spec_data, wave_data) 
            
            _, preds = torch.max(outputs, 1)
            
            acc.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = acc.compute().item()
    test_precision = precision.compute().item()
    test_recall = recall.compute().item()

    print("\n--- Risultati Sintetici (Multi-Stream) ---")
    print(f" Test Accuracy: {test_accuracy:.4f}")
    print(f" Test Macro Precision: {test_precision:.4f}")
    print(f" Test Macro Recall: {test_recall:.4f}")

    print("\n*** Classification Report")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names) 
    plt.xlabel('Predizione')
    plt.ylabel('Verità (Target)')
    plt.title('Matrice di Confusione (Test Set Multi-Stream)')
    plt.show()

    return test_accuracy

if __name__ == '__main__':
    set_seed(42)
    train_loader, validation_loader, test_loader, class_names = get_dataloaders(
        spec_train_dir=spec_train_dir, wave_train_dir=wave_train_dir,
        spec_test_dir=spec_test_dir, wave_test_dir=wave_test_dir,
        spec_transforms=standard_transforms, wave_transforms=standard_transforms,
        batch_size=BATCH_SIZE, val_split_ratio=VAL_SPLIT_RATIO
    )
    
    model = MultiStreamResNet(NUM_CLASSES, DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_acc = 0.0

    print("\n*** Starting training")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for spec_data, wave_data, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            spec_data, wave_data, targets = spec_data.to(DEVICE), wave_data.to(DEVICE), targets.to(DEVICE)
            
            scores = model(spec_data, wave_data) 
            loss = criterion(scores, targets)
            
            optimizer.zero_grad(); loss.backward(); optimizer.step();
            epoch_loss += loss.item() * targets.size(0)

        total_train_loss = epoch_loss / len(train_loader.dataset)
        val_acc = validate_epoch(model, validation_loader, DEVICE)
        
        print(f"Loss: {total_train_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f">>> New record: {best_val_acc:.4f}")

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    test_model(model, test_loader, DEVICE, class_names)

    print("\nTraining completato. Eseguire la valutazione finale...")

