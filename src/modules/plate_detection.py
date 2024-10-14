import cv2

def detect_plate(image_path, cascade_path):
    plate_cascade = cv2.CascadeClassifier(cascade_path)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image at {image_path} not found.")
        return None, None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(plates) == 0:
        return image, None
    
    x, y, w, h = plates[0]
    plate_image = image[y:y+h, x:x+w]
    return image, plate_image
