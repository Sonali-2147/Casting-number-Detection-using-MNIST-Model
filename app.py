import cv2
import tensorflow as tf
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 model 
model = YOLO('yolov8n.pt')

# Load MNIST model 
mnist_model = tf.keras.models.load_model('mnist_final_model.keras')

# Preprocess image for MNIST model 
def preprocess_for_mnist(detected_region):
    detected_region_gray = cv2.cvtColor(detected_region, cv2.COLOR_BGR2GRAY)  
    detected_region_resized = cv2.resize(detected_region_gray, (28, 28))  
    detected_region_resized = detected_region_resized.astype('float32') / 255.0  
    detected_region_resized = np.reshape(detected_region_resized, (1, 28, 28, 1))  
    return detected_region_resized

# Extract detected regions and recognize numbers using MNIST model
def extract_and_recognize_numbers_mnist(image, results, box_reduction_factor=0.1, min_difference=5):
    output_images = [] 
    bounding_boxes = [] 
    recognized_numbers = [] 

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            confidence = box.conf[0] 

            
            if confidence > 0.3:  
                width = x2 - x1
                height = y2 - y1
                x1 = int(x1 + box_reduction_factor * width)
                y1 = int(y1 + box_reduction_factor * height)
                x2 = int(x2 - box_reduction_factor * width)
                y2 = int(y2 - box_reduction_factor * height)

                detected_region = image[y1:y2, x1:x2] 
                
                
                preprocessed_region = preprocess_for_mnist(detected_region)
                
                
                prediction = mnist_model.predict(preprocessed_region)
                predicted_number = np.argmax(prediction)  

                recognized_numbers.append((predicted_number, (x1, y1, x2, y2)))
                output_images.append(detected_region)
                bounding_boxes.append((x1, y1, x2, y2, predicted_number)) 

    return output_images, bounding_boxes

# Load input image
image_path = 'image 2.png'  
image = cv2.imread(image_path)

# Resize the image to match the expected input size for the YOLO model
image_resized = cv2.resize(image, (640, 640))

# Perform detection using YOLO
results = model(image_resized)

# Extract regions containing numbers and recognize text using MNIST model
detected_regions, bounding_boxes = extract_and_recognize_numbers_mnist(image, results, box_reduction_factor=0.1, min_difference=5)

# Draw bounding boxes around the detected regions with predictions
for (x1, y1, x2, y2, number) in bounding_boxes:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"Identified Number: {number}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the result
cv2.imshow("Detected Numbers", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the output image with bounding boxes and predicted numbers
cv2.imwrite("output_image_with_boxes_and_predictions.png", image)