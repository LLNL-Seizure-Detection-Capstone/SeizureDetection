#==========================================================================================================#
# MODELS.PY
# This file is meant to hold all structures of different Machine Learning Models
#==========================================================================================================#
import torch
import warnings

# Ignore User warning about the shaping a even kernel with an odd dialation
warnings.filterwarnings("ignore")

# Architecture For 2D-CNN AE + MLP is found under "A Deep Learning Approach for Automatic Seizure Detection in Children With Epilepsy"
# ASSUMING TIME STEP of 2 seconds
# BINARY CLASSIFIER
class CNN_AE_MLP(torch.nn.Module) :
    def __init__(self, **kwargs) :
        super().__init__()
        # BATCH X CHANNEL X HEIGHT X WIDTH
        # BATCH X CHANNEL X TIME SERIES X ELECTRODE NODE
        self.encoder = torch.nn.Sequential(
            # BATCH X 1 X 512 X 24
            torch.nn.Conv2d(1, 32, (3,2), padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, (3,2), padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, (3,2), padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 64, (3,2), padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
            # 32 X 1 X 64
        )

        self.decoder = torch.nn.Sequential(
            # 32 X 1 X 64
            torch.nn.Upsample(size=(64, 3)),
            torch.nn.Conv2d(64, 64, (3,2), padding='same'),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(64, 32, (3,2), padding='same'),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(32, 32, (3,2), padding='same'),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(32, 1, (3,2), padding='same'),
            # 512 X 24 X 1
         )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 50),
            torch.nn.ReLU(),
            torch.nn.Dropout(.2), # Added from original architecture
            torch.nn.Linear(50, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(.2), # Added from original architecture
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
         )

    # Returns the decoded information first and then returns the label
    def forward(self, x) :
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        out = self.classifier(encoded)
        return decoded, out

    def freeze_autoencoder(self) :
        # Freeze the AE Layers to prevent them from training
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters() :
            param.requires_grad = False
        return

