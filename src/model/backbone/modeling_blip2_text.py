import torch
from torch import nn
from transformers import (
    AutoConfig, 
    AutoModelForSeq2SeqLM, 
    PreTrainedModel, 
    Blip2QFormerModel, 
    Blip2Config
)
from typing import Optional, Union, Tuple
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PrefixTuningConfig, PeftModel

class CustomBLIP2Seq2SeqLM(PreTrainedModel):
    def __init__(self,
                 vit_pretrained="google/vit-base-patch16-224-in21k",
                 qformer_pretrained="Salesforce/blip2-opt-2.7b",
                 lm_pretrained="VietAI/vit5-base",
                 freeze_qformer=False,
                 num_query_token=64,
                 freeze_lm=True,
                 use_lora=False,
                 cast_dtype=torch.float32,
                 lora_alpha=16,
                 lora_r=8,
                 lora_dropout=0.05,
                 lora_bias="none",
                 prefix_tokens=32):
        # Initialize configuration
        config = Blip2Config.from_pretrained(qformer_pretrained)
        config.num_query_tokens = num_query_token
        vision_config = AutoConfig.from_pretrained(vit_pretrained)
        lm_config = AutoConfig.from_pretrained(lm_pretrained)
        super().__init__(config)
        
        # Load language model
        self.language_model = AutoModelForSeq2SeqLM.from_pretrained(lm_pretrained, config=lm_config)
        lm_config.bos_token_id = lm_config.decoder_start_token_id
        lm_hidden_size = lm_config.d_model
        config.text_config = lm_config
        config.vision_config = vision_config

        # Set up projections and query tokens
        self.vision_projection = nn.Linear(vision_config.hidden_size, config.qformer_config.encoder_hidden_size)
        self.qformer = Blip2QFormerModel.from_pretrained(qformer_pretrained, config=config.qformer_config)
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, lm_hidden_size)
        self.llm_cast_dtype = cast_dtype

        # Freeze language model if required
        if freeze_lm:
            for param in self.language_model.parameters():
                param.requires_grad = False

        # Freeze QFormer if required
        if freeze_qformer:
            # self.query_tokens.requires_grad = False
            for param in self.qformer.parameters():
                param.requires_grad = False

        # Set up LoRA if enabled
        if use_lora and not freeze_lm:
            print("Using LoRA")
            self.language_model = prepare_model_for_kbit_training(self.language_model, use_gradient_checkpointing=True)
            if isinstance(use_lora, bool) or use_lora=="lora":
                target_modules=["q", "v"]
                config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias=lora_bias,
                    task_type='SEQ_2_SEQ_LM'
                )
            elif use_lora == "lora_all":
                target_modules = ["q", ".k", "v", ".o", "wi_0", "wi_1", "wo"]
                config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias=lora_bias,
                    task_type='SEQ_2_SEQ_LM'
                )
            elif use_lora == "prefix":
                config = PrefixTuningConfig(
                    task_type='SEQ_2_SEQ_LM',
                    num_virtual_tokens=prefix_tokens,
                )
            self.language_model = get_peft_model(self.language_model, config)

    def forward(self, 
                visual_features: torch.FloatTensor, 
                ocr_features: Optional[torch.FloatTensor] = None,
                input_ids: Optional[torch.FloatTensor] = None, 
                attention_mask: Optional[torch.LongTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None, 
                decoder_attention_mask: Optional[torch.LongTensor] = None,
                labels: Optional[torch.LongTensor] = None, 
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None, 
                return_dict: Optional[bool] = None,
                **kwargs) -> Union[Tuple, dict]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Concatenate OCR features if present
        if ocr_features is not None:
            visual_features = torch.cat([visual_features, ocr_features], dim=1)

        # Process visual features
        image_embeds = self.vision_projection(visual_features)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(query_embeds=query_tokens, encoder_hidden_states=image_embeds,
                                     encoder_attention_mask=torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device),
                                     return_dict=return_dict)
        query_output = query_outputs[0]
        language_model_inputs = self.language_projection(query_output)

        # Prepare inputs for language model
        if input_ids is not None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([language_model_inputs, inputs_embeds], dim=1)
            attention_mask = torch.cat([torch.ones(language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device), attention_mask], dim=1)
        else:
            input_ids = torch.LongTensor([[self.config.text_config.bos_token_id]]).repeat(image_embeds.shape[0], 1).to(image_embeds.device)
            inputs_embeds = torch.cat([language_model_inputs, self.language_model.get_input_embeddings()(input_ids)], dim=1)
            attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=inputs_embeds.device)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask,
            labels=labels, return_dict=return_dict
        )

        if return_dict:
            return {"loss": outputs.loss, "logits": outputs.logits}
        else:
            return outputs

    @torch.no_grad()
    def generate(self, 
                 visual_features: torch.FloatTensor, 
                 ocr_features: Optional[torch.FloatTensor] = None,
                 input_ids: Optional[torch.LongTensor] = None, 
                 attention_mask: Optional[torch.LongTensor] = None,
                 **generate_kwargs) -> torch.LongTensor:
        if ocr_features is not None:
            visual_features = torch.cat([visual_features, ocr_features], dim=1)

        image_embeds = self.vision_projection(visual_features)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(query_embeds=query_tokens, encoder_hidden_states=image_embeds,
                                     encoder_attention_mask=torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device),
                                     return_dict=True)
        language_model_inputs = self.language_projection(query_outputs.last_hidden_state)

        if input_ids is not None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([language_model_inputs, inputs_embeds], dim=1)
            attention_mask = torch.cat([torch.ones(language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device), attention_mask], dim=1)
        else:
            input_ids = torch.LongTensor([[self.config.text_config.bos_token_id]]).repeat(image_embeds.shape[0], 1).to(image_embeds.device)
            inputs_embeds = torch.cat([language_model_inputs, self.language_model.get_input_embeddings()(input_ids)], dim=1)
            attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=inputs_embeds.device)

        return self.language_model.generate(inputs_embeds=inputs_embeds, 
                                            attention_mask=attention_mask, 
                                            **generate_kwargs)
    
    def _preprocess_accelerate(self):
        hf_device_map = self.hf_device_map

        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            print("Warning: The `language_model` is not in `hf_device_map`, may cause issues in multi-GPU environments.")
        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # For compatibility during generation

