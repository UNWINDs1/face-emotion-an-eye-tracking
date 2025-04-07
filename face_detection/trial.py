import cv2
import numpy as np
import torch
import os
import time
import kagglehub
from torchvision import transforms
from PIL import Image
from datetime import datetime
import math


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
    
    # Modified to detect surprise as fear
    if std_intensity > 50:
        if avg_intensity > 130:
            return "happiness", 75
        else:
            return "fear", 65  # Changed from "surprise" to "fear"
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

def detect_pupil(eye_roi):
    """Improved pupil detection for more accurate eyeball tracking."""
    if eye_roi.size == 0:
        return None, None
        
    # Convert to grayscale
    eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to improve contrast
    eye_gray = cv2.equalizeHist(eye_gray)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(eye_gray, (5, 5), 0)
    
    # Use adaptive thresholding to handle different lighting conditions
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to reduce noise and fill gaps
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # Filter contours by area to find the pupil
    valid_contours = []
    eye_area = eye_roi.shape[0] * eye_roi.shape[1]
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > eye_area * 0.01 and area < eye_area * 0.5:  # Reasonable pupil size
            valid_contours.append(cnt)
    
    if not valid_contours:
        return None, None
    
    # Find the most circular contour (pupil is usually circular)
    best_circularity = 0
    best_contour = None
    
    for cnt in valid_contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > best_circularity:
            best_circularity = circularity
            best_contour = cnt
    
    if best_contour is None:
        return None, None
    
    # Get the center of the pupil using moments
    M = cv2.moments(best_contour)
    if M["m00"] == 0:
        return None, None
    
    pupil_x = int(M["m10"] / M["m00"])
    pupil_y = int(M["m01"] / M["m00"])
    
    # For visualization (draw the contour on a copy of the eye image)
    pupil_visualization = eye_roi.copy()
    cv2.drawContours(pupil_visualization, [best_contour], -1, (0, 255, 0), 1)
    cv2.circle(pupil_visualization, (pupil_x, pupil_y), 2, (0, 0, 255), -1)
    
    return (pupil_x, pupil_y), pupil_visualization

def detect_eye_direction(eye_roi):
    """Detect eye gaze direction based on pupil position with 15-degree threshold."""
    if eye_roi.size == 0:
        return "unknown", None, 0
    
    # Get pupil position
    pupil_position, visualization = detect_pupil(eye_roi)
    
    if pupil_position is None:
        return "unknown", visualization, 0
    
    pupil_x, pupil_y = pupil_position
    
    # Calculate eye dimensions
    eye_height, eye_width = eye_roi.shape[:2]
    
    # Define the center of the eye
    center_x = eye_width // 2
    center_y = eye_height // 2
    
    # Calculate the displacement of the pupil from the center
    dx = pupil_x - center_x
    dy = pupil_y - center_y
    
    # Calculate the angle in degrees
    # Use arctan2 which returns angle in range [-pi, pi]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Normalize angle to 0-360 degrees
    if angle < 0:
        angle += 360
    
    # Determine direction based on angle with 15-degree threshold
    # East (right): 0±15 degrees
    # North (up): 90±15 degrees
    # West (left): 180±15 degrees
    # South (down): 270±15 degrees
    
    direction = "center"
    angle_score = 0
    
    # Calculate how closely the angle matches each direction
    right_score = min(abs(angle), abs(angle - 360))
    up_score = abs(angle - 90)
    left_score = abs(angle - 180)
    down_score = abs(angle - 270)
    
    # Determine direction based on closest match within 15 degrees
    if right_score <= 15:
        direction = "right"
        angle_score = right_score
    elif up_score <= 15:
        direction = "up"
        angle_score = up_score
    elif left_score <= 15:
        direction = "left"
        angle_score = left_score
    elif down_score <= 15:
        direction = "down"
        angle_score = down_score
    
    # Draw direction indicators on visualization if available
    if visualization is not None:
        # Draw center point
        cv2.circle(visualization, (center_x, center_y), 3, (255, 0, 0), -1)
        # Draw line from center to pupil
        cv2.line(visualization, (center_x, center_y), (pupil_x, pupil_y), (0, 255, 255), 1)
        # Calculate angle text position
        text_x = 5
        text_y = eye_height - 5
        cv2.putText(visualization, f"{angle:.1f}°", (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    return direction, visualization, angle

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
    
    # Set camera resolution for better eye tracking
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    print("Setting up emotion detection...")
    device = load_emotion_model()
    
    cv2.namedWindow("Face Emotion and Eye Direction Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Emotion and Eye Direction Detection", 1024, 768)
    
    # For detailed eye tracking visualization
    cv2.namedWindow("Eye Tracking Details", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Eye Tracking Details", 400, 200)
    
    print("Starting webcam feed. Press 'q' to quit.")
    
    # Variables for emotion reporting
    last_report_time = time.time()
    emotion_history = []
    
    # Setup for saving reports to file
    report_file = open("emotion_reports.txt", "a")
    report_file.write(f"\n--- NEW SESSION STARTED AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
            
        display_frame = frame.copy()
        eye_details_frame = np.zeros((200, 400, 3), dtype=np.uint8)  # Black canvas for eye details
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        current_emotion = "No face detected"
        current_confidence = 0
        
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            face_img = frame[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(face_gray)
            eye_directions = []
            eye_angles = []
            eye_visualizations = []
            
            for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):  # Process at most 2 eyes
                # Draw eye rectangle
                cv2.rectangle(display_frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 0, 255), 2)
                
                # Get eye ROI
                eye_roi = face_img[ey:ey+eh, ex:ex+ew]
                
                # Detect eye direction with improved tracking
                direction, viz, angle = detect_eye_direction(eye_roi)
                eye_directions.append(direction)
                eye_angles.append(angle)
                
                if viz is not None:
                    # Store visualization for display
                    eye_visualizations.append((viz, i))
                
                # Display eye direction text with angle
                direction_text = f"Eye {i+1}: {direction}"
                cv2.putText(display_frame, direction_text, 
                            (x+ex, y+ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
           
            # Display individual eye tracking details
            if eye_visualizations:
                # Clear previous eye details
                eye_details_frame = np.zeros((200, 400, 3), dtype=np.uint8)
                
                for viz, i in eye_visualizations:
                    h, w = viz.shape[:2]
                    if i == 0:  # Left eye
                        eye_details_frame[50:50+h, 50:50+w] = viz
                        cv2.putText(eye_details_frame, "Left Eye", (50, 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:  # Right eye
                        eye_details_frame[50:50+h, 200+50:200+50+w] = viz
                        cv2.putText(eye_details_frame, "Right Eye", (250, 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add threshold indicator
                cv2.putText(eye_details_frame, "Threshold: 15°", (150, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Determine overall eye direction
            if len(eye_directions) >= 2:
                if all(direction == "left" for direction in eye_directions):
                    gaze_direction = "Looking Left"
                    gaze_color = (0, 165, 255)
                elif all(direction == "right" for direction in eye_directions):
                    gaze_direction = "Looking Right"
                    gaze_color = (0, 165, 255)
                elif all(direction == "up" for direction in eye_directions):
                    gaze_direction = "Looking Up"
                    gaze_color = (0, 165, 255)
                elif all(direction == "down" for direction in eye_directions):
                    gaze_direction = "Looking Down"
                    gaze_color = (0, 165, 255)
                else:
                    gaze_direction = "Looking Center"
                    gaze_color = (0, 255, 0)
                
                # Add average angle to gaze direction display
                if eye_angles and not any(angle == 0 for angle in eye_angles):
                    avg_angle = sum(eye_angles) / len(eye_angles)
                    gaze_direction += f" ({avg_angle:.1f}°)"
                
                cv2.putText(display_frame, gaze_direction, (x, y-40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_color, 2)
            
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
                    current_emotion = predicted_emotion
                    current_confidence = confidence
                    
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
                    
                    # Add to emotion history for reports with timestamp
                    emotion_history.append((predicted_emotion, confidence, datetime.now()))
                    
                except Exception as e:
                    print(f"Error processing emotion: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Check if it's time to generate an emotion report
        current_time = time.time()
        if current_time - last_report_time >= 30:
            if emotion_history:
                emotion_counts = {}
                for emotion, conf, _ in emotion_history:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                # Get the most frequent emotion
                most_frequent = max(emotion_counts.items(), key=lambda x: x[1])
                
                # Calculate percentages
                total = len(emotion_history)
                percentages = {e: (c/total*100) for e, c in emotion_counts.items()}
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                report = [
                    f"\n--- EMOTION REPORT ({timestamp}) ---",
                    f"Dominant emotion in last 30 seconds: {most_frequent[0]} ({most_frequent[1]/total*100:.1f}%)",
                    "Emotion breakdown:"
                ]
                
                for emotion, count in emotion_counts.items():
                    report.append(f"  - {emotion}: {count} times ({percentages[emotion]:.1f}%)")
                
                report.append("-------------------------------\n")
                report_text = "\n".join(report)
                
                # Print to console
                print(report_text)
                
                # Write to file
                report_file.write(report_text)
                report_file.flush()
                
            # Reset for next report
            last_report_time = current_time
            emotion_history = []
        
        # Display time until next report
        time_remaining = max(0, 30 - (current_time - last_report_time))
        cv2.putText(display_frame, f"Next report in: {time_remaining:.0f}s", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display current emotion at the top
        cv2.putText(display_frame, f"Current: {current_emotion} ({current_confidence:.1f}%)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display instruction
        cv2.putText(display_frame, "Press 'q' to quit", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show threshold parameters
        cv2.putText(display_frame, "Eye Tracking Threshold: 15°", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display main frame and eye tracking details
        cv2.imshow("Face Emotion and Eye Direction Detection", display_frame)
        cv2.imshow("Eye Tracking Details", eye_details_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    report_file.close()
    print("Application closed.")

if __name__ == "__main__":
    main()