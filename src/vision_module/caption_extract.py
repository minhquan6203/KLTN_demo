import os
import json
from tqdm import tqdm
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

class CaptionExtractor:
    def __init__(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            'Qwen/Qwen2-VL-2B-Instruct',
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # attn_implementation="flash_attention_2",
        )
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoProcessor.from_pretrained(
            'Qwen/Qwen2-VL-2B-Instruct',
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )

    def conversation_template(self, questions, image_paths):
        messages = []
        for question, image_path in zip(questions, image_paths):
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {
                            "type": "text",
                            "text": (
                                f"Hãy quan sát kỹ hình ảnh và dựa vào câu hỏi sau: '{question}'. "
                                "Nếu có văn bản xuất hiện trong hình ảnh (OCR), hãy sử dụng các thông tin từ văn bản này để trả lời câu hỏi như một caption mô tả chi tiết hình ảnh. "
                                "Hãy mô tả rõ ràng từng chi tiết trong văn bản và cách nó liên quan đến bối cảnh chung của hình ảnh. "
                                "Nếu không có văn bản hoặc văn bản không có liên quan, hãy mô tả đầy đủ mọi yếu tố khác của bức ảnh như bối cảnh, đối tượng, màu sắc, bố cục, và những chi tiết nhỏ mà bạn có thể nhìn thấy. "
                                "Hãy sử dụng ngôn ngữ phong phú và mở rộng mô tả của bạn để tạo ra một câu trả lời dài và chi tiết nhất có thể. Liên hệ các yếu tố trong hình ảnh với ngữ cảnh hoặc giả thuyết về những gì đang diễn ra trong hình."
                            )
                        }
                    ]
                }
            ]
            messages.append(conversation)
        return messages

    def get_caption(self, question, image_path):
        self.model.eval()
        with torch.no_grad():
            messages = self.conversation_template([question], [image_path])
            image_inputs, video_inputs = process_vision_info(messages)
            text_prompt = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            inputs = self.processor(text=text_prompt, images=image_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to(self.device)

            output_ids = self.model.generate(**inputs, max_new_tokens=256)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            del inputs
            del output_ids
            torch.cuda.empty_cache()
            return output_texts[0]
