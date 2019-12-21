"""
    @author lfko
    @summary Using a CNN for text classification 
    @NB: - https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf
         - https://github.com/uvipen/Character-level-cnn-pytorch
         - https://github.com/srviest/char-cnn-text-classification-pytorch
         - https://www.kaggle.com/bittlingmayer/amazonreviews
"""

# PyTorch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data
import torch.optim as optim


import numpy as np

class TextCNN(nn.Module):
    def __init__(self, num_classes = 2):
        super(TextCNN, self).__init__()

        # L1: Conv, ReLU 
        # L2: Max-Pool
        # L3: Conv, ReLU
        # L4: Max-Pool
        # L5: Conv, ReLU
        # L6: Conv, ReLU
        # L7: Conv, ReLU
        # L8: Conv, ReLU
        # L9: Max-Pool
        # L10: Flatten
        # L11/12/13: FC, FC, FC

        self.layer1 = nn.Sequential(
            nn.Conv2d(70, 1024, 7, 1), nn.ReLU() 
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 256, 7, 1), nn.ReLU()
        )

        self.layer5678 = nn.Sequential(
            nn.Conv2d(64, 64, 7, 1), nn.ReLU()
        )

        self.max_pool_layer = nn.Sequential(
            nn.MaxPool2d(3)
        )

        self.fc = nn.Sequential(
            #nn.Linear(64, 2048), nn.ReLU(), nn.Dropout2d(.5)
            #nn.Linear(2048, 2048), nn.ReLU(), nn.Dropout2d(.5)
            nn.Linear(64, 2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.ReLU()
        )

        self.out = nn.Sequential(
            nn.Linear(2048, 2), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.max_pool_layer(x)
        x = self.layer3(x)
        x = self.max_pool_layer(x)
        x = self.layer5678(x)
        x = x.view(-1, x.size(0)) # Flatten
        x = self.fc(x)

        return self.out(x)