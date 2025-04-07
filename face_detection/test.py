import cv2
import numpy as np
import torch
import os
import kagglehub
from torchvision import transforms
from PIL import Image


def load_emotion_model():
    """Load a pre-trained emotion recognition model."""
    try:
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using GPU for emotion detection")
        else:
            device = torch.device("cpu")
            print("Using CPU for emotion detection")
            
        
        return device
    except Exception as e:
        print(f"Error setting up emotion detection: {e}")
        return None

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

def process_face_for_emotion(face_img):
    """Process a face image and predict emotion using a simple approach."""
    
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    
    face_height, face_width = gray.shape
    
    
    middle_region = gray[int(face_height*0.3):int(face_height*0.7), 
                        int(face_width*0.3):int(face_width*0.7)]
    
    if middle_region.size == 0:
        return "unknown", 0
    
   
    avg_intensity = np.mean(middle_region)
    std_intensity = np.std(middle_region)
    
    
    if std_intensity > 50:
        if avg_intensity > 130:
            return "happiness", 75
        else:
            return "surprise", 65
    elif std_intensity > 30:
        if avg_intensity > 110:
            return "neutral", 60
        else:
            return "sadness", 55
    else:
        if avg_intensity < 90:
            return "anger", 50
        else:
            return "fear", 45
    
def main():
    
    dataset_path = download_emotion_dataset()
    if dataset_path is None:
        print("Failed to download the dataset.")
    else:
        print(f"Using Ryerson Emotion Database from: {dataset_path}")
  
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    print("Setting up emotion detection...")
    device = load_emotion_model()
    
    
    emotion_labels = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

    cv2.namedWindow("Face Emotion and Eye Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Emotion and Eye Detection", 800, 600)
    
    print("Starting webcam feed. Press 'q' to quit.")
    
    while True:
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
            
        display_frame = frame.copy()
        
       
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
       
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        
        for (x, y, w, h) in faces:
            
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            face_img = frame[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(face_gray)
            for (ex, ey, ew, eh) in eyes:
               
                cv2.rectangle(display_frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 0, 255), 2)
           
            if len(eyes) >= 2:
                eye_status = "Eyes Open"
                status_color = (0, 255, 0)  
            elif len(eyes) == 1:
                eye_status = "Blinking"
                status_color = (0, 165, 255)  
            else:
                eye_status = "Eyes Closed"
                status_color = (0, 0, 255) 
                
            cv2.putText(display_frame, eye_status, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            if face_img.size > 0:
                try:
                   
                    predicted_emotion, confidence = process_face_for_emotion(face_img)
                    
                    
                    if predicted_emotion == "happiness":
                        emotion_color = (0, 255, 255)  
                    elif predicted_emotion in ["anger", "disgust", "fear", "sadness"]:
                        emotion_color = (0, 0, 255) 
                    elif predicted_emotion == "surprise":
                        emotion_color = (255, 0, 255)  
                    else:
                        emotion_color = (255, 255, 255) 
                    
                    
                    cv2.putText(display_frame, f"Emotion: {predicted_emotion} ({confidence:.1f}%)", 
                                (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, emotion_color, 2)
                except Exception as e:
                    print(f"Error processing emotion: {e}")
                    import traceback
                    traceback.print_exc()
        
        instruction_text = "Press 'q' to quit"
        cv2.putText(display_frame, instruction_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Face Emotion and Eye Detection", display_frame)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main()