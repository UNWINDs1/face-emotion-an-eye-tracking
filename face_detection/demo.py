import cv2
import numpy as np
import torch
from PIL import Image
import os
import kagglehub
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

def download_emotion_dataset():
    """Download the Ryerson Emotion Database from Kaggle."""
    try:
        print("Downloading Ryerson Emotion Database...")
        path = kagglehub.dataset_download("ryersonmultimedialab/ryerson-emotion-database")
        print(f"Dataset downloaded successfully to: {path}")
        return path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def load_emotion_model():
    """Load the face emotion recognition model from HuggingFace."""
    try:
        model_name = "ElenaRyumina/face_emotion_recognition"
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        return feature_extractor, model
    except Exception as e:
        print(f"Error loading emotion model: {e}")
        print(f"Detailed error: {str(e)}")
        return None, None

def main():
    # Download the Ryerson Emotion Database
    dataset_path = download_emotion_dataset()
    if dataset_path is None:
        print("Failed to download the dataset. Using default emotion model.")
    else:
        print(f"Using Ryerson Emotion Database from: {dataset_path}")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Load cascade classifiers for face and eyes
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    print("Loading emotion recognition model...")
    # Load emotion recognition model
    feature_extractor, emotion_model = load_emotion_model()
    if feature_extractor is None or emotion_model is None:
        print("Failed to load emotion model!")
    else:
        print("Emotion model loaded successfully!")
    
    # Define emotion labels (matching Ryerson database categories)
    emotion_labels = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
    
    # Window setup
    cv2.namedWindow("Face Emotion and Eye Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Emotion and Eye Detection", 800, 600)
    
    print("Starting webcam feed. Press 'q' to quit.")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
            
        # Make a copy of the frame for displaying results
        display_frame = frame.copy()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Crop the face
            face_img = frame[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]
            
            # Detect eyes within the face region
            eyes = eye_cascade.detectMultiScale(face_gray)
            for (ex, ey, ew, eh) in eyes:
                # Draw rectangle around eyes
                cv2.rectangle(display_frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 0, 255), 2)
            
            # Display eye status
            if len(eyes) >= 2:
                eye_status = "Eyes Open"
                status_color = (0, 255, 0)  # Green
            elif len(eyes) == 1:
                eye_status = "Blinking"
                status_color = (0, 165, 255)  # Orange
            else:
                eye_status = "Eyes Closed"
                status_color = (0, 0, 255)  # Red
                
            cv2.putText(display_frame, eye_status, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Process emotion if model is loaded and face is detected
            if emotion_model is not None and feature_extractor is not None and face_img.size > 0:
                try:
                    # Convert BGR to RGB (PIL expects RGB)
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image
                    pil_image = Image.fromarray(face_rgb)
                    
                    # Use feature extractor
                    inputs = feature_extractor(images=pil_image, return_tensors="pt")
                    
                    # Get emotion prediction
                    with torch.no_grad():
                        outputs = emotion_model(**inputs)
                    
                    # Get the predicted class
                    predicted_class_idx = outputs.logits.argmax(-1).item()
                    
                    # Get confidence score
                    confidence = torch.nn.functional.softmax(outputs.logits, dim=1)[0][predicted_class_idx].item() * 100
                    
                    # Map to emotion label (ensure index is in range)
                    if predicted_class_idx < len(emotion_labels):
                        predicted_emotion = emotion_labels[predicted_class_idx]
                    else:
                        predicted_emotion = "unknown"
                    
                    # Choose color based on emotion
                    if predicted_emotion == "happiness":
                        emotion_color = (0, 255, 255)  # Yellow
                    elif predicted_emotion in ["anger", "disgust", "fear", "sadness"]:
                        emotion_color = (0, 0, 255)  # Red
                    elif predicted_emotion == "surprise":
                        emotion_color = (255, 0, 255)  # Purple
                    else:
                        emotion_color = (255, 255, 255)  # White
                    
                    # Display emotion on the frame
                    cv2.putText(display_frame, f"Emotion: {predicted_emotion} ({confidence:.1f}%)", 
                                (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, emotion_color, 2)
                except Exception as e:
                    print(f"Error processing emotion: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Add instructions to the display
        instruction_text = "Press 'q' to quit"
        cv2.putText(display_frame, instruction_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the resulting frame
        cv2.imshow("Face Emotion and Eye Detection", display_frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main()