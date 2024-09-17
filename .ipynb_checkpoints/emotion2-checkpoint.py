import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = tf.keras.models.load_model('my_model.keras')

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a function to preprocess the image
def preprocess_image(img, target_size=(256, 256)):
    img = cv2.resize(img, target_size)  # Resize image to target size
    img_array = np.array(img)           # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0       # Normalize the image
    return img_array

# Define a function to predict emotion
def predict_emotion(img_array):
    predictions = model.predict(img_array)
    return 'Happy' if predictions[0][0] > 0.5 else 'Sad'

# Initialize camera
cap = cv2.VideoCapture(0)  # 0 is the default camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract the face region
        face_region = frame[y:y+h, x:x+w]
        
