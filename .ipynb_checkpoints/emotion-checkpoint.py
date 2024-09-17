import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = tf.keras.models.load_model('path_to_your_model.h5')

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
    
    # Preprocess the frame
    img_array = preprocess_image(frame)
    
    # Predict emotion
    emotion = predict_emotion(img_array)
    
    # Display the resulting frame
    cv2.putText(frame, f'Emotion: {emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Emotion Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
