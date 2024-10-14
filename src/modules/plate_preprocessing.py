import cv2

def preprocess_plate(plate_image):
    # Convert to grayscale
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adjust contrast and brightness (optional but effective for OCR)
    alpha = 1.5  # Contrast control
    beta = 30    # Brightness control
    adjusted = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)
    
    # Apply adaptive thresholding to handle varying lighting conditions
    adaptive_thresh = cv2.adaptiveThreshold(adjusted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
    cv2.imshow('image adaptive_thresh', adaptive_thresh)
    return adaptive_thresh
