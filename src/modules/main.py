import os
import cv2
from plate_detection import detect_plate
from plate_preprocessing import preprocess_plate
from plate_ocr import recognize_text

# Define the root directory of the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths relative to the root directory
image_path = os.path.join(ROOT_DIR, 'images', 'carro1.jpg')
cascade_path = os.path.join(ROOT_DIR, 'haarcascade_russian_plate_number.xml')

# Detect plate
original_image, plate_image = detect_plate(image_path, cascade_path)

if plate_image is not None:
    # Preprocess plate
    preprocessed_image = preprocess_plate(plate_image)
    
    # Recognize text
    text = recognize_text(preprocessed_image)
    print(f"Recognized License Plate Text: {text}")
else:
    print("No plate detected")

# Display the results
cv2.imshow('Original Image', original_image)
if plate_image is not None:
    cv2.imshow('Plate Image', plate_image)
    cv2.imshow('Preprocessed Plate', preprocessed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
