import pytesseract

def recognize_text(preprocessed_plate):
    custom_config = r'--oem 1 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(preprocessed_plate, config=custom_config)
    return text.strip()
