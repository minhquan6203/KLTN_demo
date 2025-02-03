import torch
from torch import nn
import os
from PIL import Image
from typing import List
from transformers import AutoFeatureExtractor, AutoModel
from collections import Counter
from typing import List, Dict, Optional,Any
import numpy as np
from vision_module.vision_pixel_encoding import VisionPixelEncoding

class VisionPixelEmbedding(nn.Module):
    def __init__(self, config: Dict) -> None:
        super(VisionPixelEmbedding,self).__init__()     
        self.visual_embedding = AutoModel.from_pretrained(config['vision_embedding']['image_encoder'])
        self.visual_encoding = VisionPixelEncoding(config)
        self.max_seq = config['vision_embedding']['max_seq']
        if config['vision_embedding']['freeze']:
            for param in self.visual_embedding.parameters():
                param.requires_grad = False

    def forward(self, image_path):
        pixels=self.visual_encoding(image_path)
        features=self.visual_embedding(pixels).last_hidden_state
        if self.max_seq is not None:
            features = features[:,:self.max_seq,:]
        return features
