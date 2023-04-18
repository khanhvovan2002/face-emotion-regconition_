from facenet_pytorch import InceptionResnetV1

import torch
import numpy as np
from torch import nn
import torchvision
import torch.nn.functional as F
import cv2
import torchvision.transforms as transforms

# Load the model state dictionary from the file

class MyModel(InceptionResnetV1):
    def __init__(self, pretrained='vggface2'):
        super(MyModel, self).__init__(pretrained=pretrained)
        
        # Replace the last fully connected layer with a new nn.Linear layer
        num_ftrs = self.logits.in_features
        self.logits = nn.Linear(num_ftrs, 7)
        
    def forward(self, x):
        # Call the forward pass of the original model
        x = super(MyModel, self).forward(x)
        
        # Call the forward pass of the last fully connected layer
        x = self.logits(x)
        x = nn.functional.softmax(x,dim=1)
        return x

