import yaml
import torch
from model.vqa_model import ViBlipVQAModel

# Load config
with open('./config/viblip_text_config.yaml') as conf_file:
    config = yaml.safe_load(conf_file)

cuda_device = config['train']['cuda_device']
device = torch.device(f'{cuda_device}' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = ViBlipVQAModel(config)
model.to(device)

def main():
    image_path = None 
    
    while True:
        use_new_image = input("Bạn có muốn tải ảnh mới không? (y/n): ").strip().lower()
        
        if use_new_image == 'y':
            image_path = input("Nhập đường dẫn ảnh mới: ").strip()
        
        if not image_path:
            print("Bạn chưa tải ảnh nào. Vui lòng tải ảnh trước khi đặt câu hỏi.")
            continue
        
        question = input("Nhập câu hỏi (hoặc gõ 'exit' để thoát): ").strip()
        if question.lower() == 'exit':
            break
        
        try:
            pred_result = model(question, image_path)
            print(f"Kết quả: {pred_result}")
        except Exception as e:
            print(f"Lỗi: {e}")

if __name__ == '__main__':
    main()