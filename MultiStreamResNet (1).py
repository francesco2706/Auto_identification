#Questo file definisce l'architettura di rete neurale che combina i due flussi di 
# dato (Spettogrammi e Waveform) --> Late Fusion Multi Modale

import torch
from torch import nn
from get_feature_extractor import get_feature_extractor # Importa dal file 1

# --- CLASSE ARCHITETTURA MULTI-STREAM ---
class MultiStreamResNet(nn.Module):
    def __init__(self, num_classes, device):
        super().__init__()
        
        print("Inizializzazione architettura Multi-Stream...")
        
        # 1. RAMO SPETTROGRAMMA (Feature Extractor, output 512)
        # Si assumono qui immagini RGB/3-canali
        self.spectrogram_extractor = get_feature_extractor(device) 
        
        # 2. RAMO WAVEFORM (Feature Extractor, output 512)
        # Si assumono qui immagini RGB/3-canali (dopo la conversione da Matplotlib)
        self.waveform_extractor = get_feature_extractor(device)
        
        # 3. LIVELLO DI FUSIONE E CLASSIFICAZIONE
        # L'input è 512 (da spec) + 512 (da wave) = 1024
        self.classifier = nn.Sequential(
            nn.Linear(512 + 512, 512), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, spec_data, wave_data):
        # Esecuzione in parallelo
        spec_features = self.spectrogram_extractor(spec_data)
        wave_features = self.waveform_extractor(wave_data)
        
        # Fusione (Concatenazione lungo la dimensione dei features)
        combined_features = torch.cat((spec_features, wave_features), dim=1)
        
        # Classificazione
        output = self.classifier(combined_features)
        return output