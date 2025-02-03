import torch
from torch import nn
import os
from PIL import Image
from typing import List
from transformers import AutoFeatureExtractor, AutoModel
from collections import Counter
from typing import List, Dict, Optional,Any
import numpy as np

class VisionPixelEncoding(nn.Module):
    def __init__(self, config: Dict):
        super(VisionPixelEncoding,self).__init__()
        self.preprocessor = AutoFeatureExtractor.from_pretrained(config["vision_embedding"]["image_encoder"])
        self.cuda_device=config['train']['cuda_device']
        self.device = torch.device(f'{self.cuda_device}' if torch.cuda.is_available() else 'cpu')
         
    def forward(self, image_path):
        processed_images = self.preprocessor(
            images=[self.load_image(image_path)],
            return_tensors="pt",
        ).to(self.device)
        return processed_images.pixel_values

    def load_image(self, image_path):
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            return image
        raise FileNotFoundError(f"Image not found for {image_path}")
