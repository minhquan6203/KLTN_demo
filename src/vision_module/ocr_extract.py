import easyocr

class OCRExtractor:
    def __init__(self):
        self.reader = easyocr.Reader(['vi'])

    def get_ocr_text(self, img_path):
        result = self.reader.readtext(img_path)
        ocr_text = ' '.join([res[1].lower() for res in result])
        return ocr_text

