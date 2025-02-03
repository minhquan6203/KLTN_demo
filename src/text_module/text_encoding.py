import torch
from torch import nn
from torch.nn import functional as F
from transformers import T5Tokenizer, AutoConfig
from typing import List, Dict, Optional
from transformers import AutoTokenizer

class ModelEncoding(nn.Module):
    def __init__(self, config: Dict):
        super(ModelEncoding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config['text_embedding']['text_encoder'])
        self.padding = config['tokenizer']['padding']
        self.max_input_length = config['tokenizer']['max_input_length']
        self.max_target_length = config['tokenizer']['max_target_length']
        self.max_scene_text = config['ocr_embedding']['max_scene_text']
        self.truncation = config['tokenizer']['truncation']
        self.cuda_device = config['train']['cuda_device']
        self.device = torch.device(f'{self.cuda_device}' if torch.cuda.is_available() else 'cpu')

    def forward(self, question: List[str], ocr_text: List[str] = None, answers: List[str] = None):
        # Encode the question
        question_encodings = self.tokenizer(question,
                                            padding=self.padding,
                                            max_length=self.max_input_length,
                                            truncation=self.truncation,
                                            return_tensors='pt').to(self.device)

        # Check and encode OCR text if provided
        if ocr_text is not None:
            ocr_encodings = self.tokenizer(ocr_text,
                                           padding=self.padding,
                                           max_length=self.max_scene_text,
                                           truncation=self.truncation,
                                           return_tensors='pt').to(self.device)
            # Concatenate question and OCR encodings
            encodings = {
                key: torch.cat([question_encodings[key], ocr_encodings[key]], dim=1)
                for key in question_encodings.keys()
            }
        else:
            encodings = question_encodings

        # If answers are provided, encode them
        if answers is not None:
            encoded_targets = self.tokenizer(answers,
                                             padding=self.padding,
                                             max_length=self.max_target_length,
                                             truncation=self.truncation,
                                             return_tensors='pt').to(self.device)

            labels_input_ids = encoded_targets["input_ids"].clone()
            decoder_attention_mask = encoded_targets["attention_mask"].clone()

            # Set pad tokens to -100 for loss calculation
            labels_input_ids[decoder_attention_mask == self.tokenizer.pad_token_id] = -100

            # Add target labels and decoder attention mask to the encodings
            encodings.update({
                'labels': labels_input_ids,
                'decoder_attention_mask': decoder_attention_mask,
            })

        return encodings
