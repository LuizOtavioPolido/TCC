import os
import cv2
from modules.plate_detection import detect_plate
from modules.plate_preprocessing import preprocess_plate
from modules.plate_ocr import recognize_text


# root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(ROOT_DIR, 'images')
image_file = 'carro4_teste.jpg'
image_path = os.path.join(image_folder, image_file)

cascade_path = os.path.join(ROOT_DIR, '..', 'haarcascade_russian_plate_number.xml')

original_image, plate_image = detect_plate(image_path, cascade_path)

if plate_image is not None:
    preprocessed_image = preprocess_plate(plate_image)
    
    cv2.imshow('preprocessed_image', preprocessed_image)

    text = recognize_text(preprocessed_image)
    print(f"placa: {text}")
else:
    print("n detectei nada fiote")

cv2.imshow('imagem original', original_image)
if plate_image is not None:
    cv2.imshow('imagem da placa', plate_image)
    cv2.imshow('imagem pre processada', preprocessed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
