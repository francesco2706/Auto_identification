import torch
from torch import nn
from torchvision import models

def get_feature_extractor(device):
    """
    Carica ResNet18 pre-addestrata e sostituisce lo strato FC con Identity 
    per usarla come estrattore di feature. L'output è un vettore 512-dimensionale.
    """
    print("Configurazione di ResNet18 come Feature Extractor...")
    
    # Scarica il modello con i pesi pre-addestrati
    weights = models.ResNet18_Weights.DEFAULT #model already learnt generic and complex features, like border, pattern and object
    model = models.resnet18(weights=weights)
    
    # Rimuove il livello Fully Connected (FC) finale per ottenere solo il vettore di feature (embedding)
    model.fc = nn.Identity()  
    
    # Nota: L'output sarà un vettore di dimensione 512
    return model.to(device)

if __name__ == '__main__':
    # Esempio di test:
    test_device = torch.device("cpu")
    extractor = get_feature_extractor(test_device)
    print("ResNet18 configurato. Output dimensione attesa:", extractor(torch.randn(1, 3, 128, 128).to(test_device)).shape)