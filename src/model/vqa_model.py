from typing import Dict
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from model.backbone.modeling_blip2_text import CustomBLIP2Seq2SeqLM
import os
from vision_module.vision_pixel_embedding import VisionPixelEmbedding
from text_module.text_encoding import ModelEncoding
from vision_module.ocr_extract import OCRExtractor
from vision_module.caption_extract import CaptionExtractor
from utils.utils import preprocess_sentence

class ViBlipVQAModel(nn.Module):
    def __init__(self, config: Dict):
        super(ViBlipVQAModel, self).__init__()
        self.processor = ModelEncoding(config)
        vision_name = config["vision_embedding"]["image_encoder"]
        lm_name = config["text_embedding"]["text_encoder"]
        freeze_lm = config["text_embedding"]["freeze"]
        qformer_name = config["qformer_embedding"]["qformer_encoder"]
        freeze_qformer = config["qformer_embedding"]["freeze"]
        num_query_tokens = config["qformer_embedding"]["num_query_tokens"]
        use_lora = config["text_embedding"]["use_lora"]
        self.save_path = os.path.join(config['train']['output_dir'], config['model']['type_model'])
        self.checkpoint_path = os.path.join(self.save_path, "best_model.pth")
        
        cast_dtype = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16
        }.get(config['train']['precision'], torch.float16)

        self.vision_embedding = VisionPixelEmbedding(config)
        self.ocr_extract = OCRExtractor()
        self.caption_extract = CaptionExtractor()

        self.embedding = CustomBLIP2Seq2SeqLM(
            vit_pretrained=vision_name,
            lm_pretrained=lm_name,
            freeze_lm=freeze_lm,
            qformer_pretrained=qformer_name,
            num_query_token=num_query_tokens,
            use_lora=use_lora,
            freeze_qformer=freeze_qformer,
            cast_dtype=cast_dtype
        )
        
        checkpoint = torch.load(self.checkpoint_path)
        new_state_dict = {k.replace("embedding.", ""): v for k, v in checkpoint['model_state_dict'].items()}
        self.embedding.load_state_dict(new_state_dict, strict=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
        self.generator_args = config["generator_args"]
        
        # Cache
        self.cache = {}

    def forward(self, question, image_path):
        if image_path in self.cache:
            visual_features, ocr_text = self.cache[image_path]
        else:
            visual_features = self.vision_embedding(image_path)
            ocr_text = self.ocr_extract.get_ocr_text(image_path)
            self.cache[image_path] = (visual_features, ocr_text)
        
        caption = self.caption_extract.get_caption(question, image_path)
        
        print('OCR text:', ocr_text)
        print('Caption text:', caption)
        
        caption = preprocess_sentence(caption)
        question = preprocess_sentence(question.replace('?', ''))
        
        inputs = self.processor([f'{question} <context> {caption}'], [ocr_text])
        inputs.update({"visual_features": visual_features})
        
        pred_ids = self.embedding.generate(**inputs, **self.generator_args)
        pred_tokens = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
        return pred_tokens
