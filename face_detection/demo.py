import cv2
import numpy as np
import os
import time
from datetime import datetime
import urllib.request
import threading
import math

# Try importing tensorflow, but provide fallback if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import model_from_json
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("TensorFlow not installed. Using simple emotion detection.")

# Try importing git, but provide fallback if not available
try:
    import git
    HAS_GIT = True
except ImportError:
    HAS_GIT = False
    print("GitPython not installed. Will download model files directly.")

def download_emotion_model():
    """Download or clone the emotion detection model."""
    if not os.path.exists("models"):
        os.makedirs("models")
        
    if HAS_GIT:
        try:
            # Clone the repository if not already present
            if not os.path.exists("Emotion-detection"):
                print("Cloning the Emotion-detection repository...")
                git.Repo.clone_from("https://github.com/atulapra/Emotion-detection", "Emotion-detection")
                print("Repository cloned successfully")
            return True
        except Exception as e:
            print(f"Error cloning repository: {e}")
            return False
    else:
        try:
            # Direct download of model files
            if not os.path.exists("models/model.json"):
                print("Downloading model architecture...")
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/atulapra/Emotion-detection/master/model/model.json",
                    "models/model.json"
                )
            
            if not os.path.exists("models/model.h5"):
                print("Downloading model weights (this may take a while)...")
                urllib.request.urlretrieve(
                    "https://github.com/atulapra/Emotion-detection/raw/master/model/model.h5",
                    "models/model.h5"
                )
            return True
        except Exception as e:
            print(f"Error downloading model files: {e}")
            return False

def download_fer_dataset():
    """Download FER2013 dataset for improved emotion detection if needed."""
    if not os.path.exists("fer_models"):
        os.makedirs("fer_models")
        
    try:
        # Download a pre-trained FER model (example URL - replace with actual model URL)
        if not os.path.exists("fer_models/fer_model.h5"):
            print("Downloading alternative FER2013 model...")
            urllib.request.urlretrieve(
                "https://github.com/priya-dwivedi/face_and_emotion_detection/raw/master/emotion_detector_models/model_v6_23.hdf5",
                "fer_models/fer_model.h5"
            )
        return True
    except Exception as e:
        print(f"Error downloading FER dataset: {e}")
        return False

def setup_emotion_model():
    """Set up emotion detection model."""
    if HAS_TENSORFLOW:
        try:
            model_path = "models/model.json"
            weights_path = "models/model.h5"
            
            if os.path.exists("Emotion-detection"):
                model_path = "Emotion-detection/model/model.json"
                weights_path = "Emotion-detection/model/model.h5"
            
            # Load model architecture
            json_file = open(model_path, 'r')
            model_json = json_file.read()
            json_file.close()
            emotion_model = model_from_json(model_json)
            
            # Load model weights
            emotion_model.load_weights(weights_path)
            print("Emotion model loaded successfully")
            
            return emotion_model
        except Exception as e:
            print(f"Error setting up emotion model: {e}")
            # Try loading alternative model
            try:
                if os.path.exists("fer_models/fer_model.h5"):
                    fer_model = tf.keras.models.load_model("fer_models/fer_model.h5")
                    print("Alternative FER model loaded successfully")
                    return fer_model
            except Exception as e2:
                print(f"Error loading alternative model: {e2}")
            return None
    else:
        return None

def detect_emotion(frame, face_coords, emotion_model):
    """Detect emotion using model if available, otherwise use a simple approach."""
    x, y, w, h = face_coords
    
    if emotion_model is not None and HAS_TENSORFLOW:
        try:
            # Preprocess the face for the model
            roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.resize(roi_gray, (48, 48))
            
            # Normalize and prepare image for model
            roi = roi_gray.astype('float')/255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)
            
            # Make prediction
            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            predictions = emotion_model.predict(roi, verbose=0)[0]  # Add verbose=0 to reduce TF output
            
            # Get the emotion with highest probability
            max_index = np.argmax(predictions)
            emotion = emotion_labels[max_index].lower()
            confidence = float(predictions[max_index]) * 100
            
            # Apply confidence threshold to reduce misclassifications
            if confidence < 40:
                # If confidence is low, use the backup method
                return improved_emotion_detection(frame, face_coords)
            
            return emotion, confidence
        except Exception as e:
            print(f"Error using TensorFlow model: {e}")
            # Fall back to improved detection
            return improved_emotion_detection(frame, face_coords)
    else:
        # Use improved detection based on image features
        return improved_emotion_detection(frame, face_coords)

def improved_emotion_detection(frame, face_coords):
    """Improved emotion detection using facial features and image processing."""
    x, y, w, h = face_coords
    face_img = frame[y:y+h, x:x+w]
    
    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_height, face_width = gray.shape
    
    # Enhanced regions of interest
    forehead_region = gray[int(face_height*0.1):int(face_height*0.3), 
                           int(face_width*0.2):int(face_width*0.8)]
    eye_region = gray[int(face_height*0.2):int(face_height*0.45), 
                      int(face_width*0.1):int(face_width*0.9)]
    mouth_region = gray[int(face_height*0.6):int(face_height*0.9), 
                        int(face_width*0.2):int(face_width*0.8)]
    
    # Handle empty regions
    if forehead_region.size == 0 or eye_region.size == 0 or mouth_region.size == 0:
        return "unknown", 0
    
    # Calculate metrics
    avg_intensity_forehead = np.mean(forehead_region)
    avg_intensity_eyes = np.mean(eye_region)
    avg_intensity_mouth = np.mean(mouth_region)
    std_intensity_mouth = np.std(mouth_region)
    
    # Calculate facial feature metrics
    brightness_diff = avg_intensity_eyes - avg_intensity_mouth
    
    # Apply Sobel operator to detect edges (captures facial expressions)
    sobelx = cv2.Sobel(mouth_region, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(mouth_region, cv2.CV_64F, 0, 1, ksize=3)
    mouth_edge_intensity = np.mean(np.sqrt(sobelx**2 + sobely**2))
    
    # Apply improved emotion classification logic
    if mouth_edge_intensity > 20 and std_intensity_mouth > 40:
        if brightness_diff > 30:
            return "happy", 80
        else:
            return "surprise", 75
    elif std_intensity_mouth > 30:
        if avg_intensity_mouth > 110:
            return "neutral", 65
        else:
            return "sad", 70
    else:
        if avg_intensity_forehead < 100:
            return "angry", 60
        elif brightness_diff < 15:
            return "fear", 55
        else:
            return "neutral", 50

def calculate_angle(eye_points, face_center):
    """Calculate the angle of gaze direction from the face center."""
    if len(eye_points) < 2:
        return 0, 0
        
    # Sort eyes by x-coordinate
    eye_points.sort(key=lambda e: e[0])
    
    # Calculate eye centroids
    left_eye = eye_points[0]
    right_eye = eye_points[1]
    
    # Calculate center point between eyes
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    
    # Calculate horizontal and vertical angles
    dx = eye_center_x - face_center[0]
    dy = eye_center_y - face_center[1]
    
    # Calculate angles in degrees
    h_angle = math.degrees(math.atan2(dx, face_center[2] / 2))  # Use face width as depth approximation
    v_angle = math.degrees(math.atan2(dy, face_center[2] / 2))
    
    return h_angle, v_angle

def analyze_eye_position(eye_points, face_width, face_height, face_x, face_y):
    """Enhanced analysis of eye gaze direction using angle-based detection."""
    if len(eye_points) < 2:
        return "unknown", 0, 0
    
    # Calculate face center
    face_center_x = face_x + face_width / 2
    face_center_y = face_y + face_height / 2
    
    # Calculate gaze angles
    h_angle, v_angle = calculate_angle(eye_points, (face_center_x, face_center_y, face_width))
    
    # Determine gaze direction based on 30-degree threshold
    if abs(h_angle) <= 30 and abs(v_angle) <= 30:
        return "looking straight", h_angle, v_angle
    
    # Determine primary direction based on which angle is larger
    if abs(h_angle) > abs(v_angle):
        # Horizontal direction is primary
        if h_angle < -30:
            h_direction = "left"
        else:  # h_angle > 30
            h_direction = "right"
            
        # Add vertical component if significant
        if abs(v_angle) > 20:
            if v_angle < 0:
                v_component = "-up"
            else:
                v_component = "-down"
        else:
            v_component = ""
        
        return f"looking {h_direction}{v_component}", h_angle, v_angle
    else:
        # Vertical direction is primary
        if v_angle < -30:
            v_direction = "up"
        else:  # v_angle > 30
            v_direction = "down"
            
        # Add horizontal component if significant
        if abs(h_angle) > 20:
            if h_angle < 0:
                h_component = "-left"
            else:
                h_component = "-right"
        else:
            h_component = ""
        
        return f"looking {v_direction}{h_component}", h_angle, v_angle

def get_emotion_icon(emotion):
    """Return a text-based emoticon based on detected emotion."""
    if emotion == "happy":
        return ":)"
    elif emotion == "sad":
        return ":("
    elif emotion == "angry":
        return ">:("
    elif emotion == "surprise":
        return ":o"
    elif emotion == "fear":
        return "D:"
    elif emotion == "disgust":
        return ":/"
    else:  # neutral or unknown
        return ":|"

def generate_emotion_report(emotion_history, gaze_history, angle_history, eyes_detected_history, report_num):
    """Generate a report of emotions and gaze during the recording period."""
    if not emotion_history:
        print("No emotion data recorded.")
        return
        
    print("\n" + "="*50)
    print(f"EMOTION AND GAZE REPORT #{report_num}")
    print("="*50)
    
    # Calculate emotion statistics
    emotion_counts = {}
    for timestamp, emotion, confidence, elapsed in emotion_history:
        if emotion not in emotion_counts:
            emotion_counts[emotion] = []
        emotion_counts[emotion].append((timestamp, confidence, elapsed))
    
    # Calculate gaze statistics
    gaze_counts = {}
    looking_away_incidents = []
    prev_gaze = None
    looking_away_start = None
    
    for timestamp, gaze, elapsed in gaze_history:
        if gaze not in gaze_counts:
            gaze_counts[gaze] = []
        gaze_counts[gaze].append((timestamp, elapsed))
        
        # Track looking away incidents with improved logic
        if ("straight" not in gaze) and prev_gaze == "looking straight":
            looking_away_start = (timestamp, elapsed)
        elif "straight" in gaze and looking_away_start is not None:
            looking_away_end = (timestamp, elapsed)
            duration = elapsed - looking_away_start[1]
            if duration > 1:  # Only count incidents longer than 1 second
                looking_away_incidents.append((looking_away_start[0], looking_away_end[0], duration, prev_gaze))
            looking_away_start = None
        
        prev_gaze = gaze
    
    # Process eyes detection statistics
    eyes_detected_count = 0
    no_eyes_detected_incidents = []
    no_eyes_start = None
    
    for timestamp, eyes_detected, elapsed in eyes_detected_history:
        if eyes_detected:
            eyes_detected_count += 1
            if no_eyes_start is not None:
                no_eyes_end = (timestamp, elapsed)
                duration = elapsed - no_eyes_start[1]
                if duration > 1:  # Only count incidents longer than 1 second
                    no_eyes_detected_incidents.append((no_eyes_start[0], timestamp, duration))
                no_eyes_start = None
        else:
            if no_eyes_start is None:
                no_eyes_start = (timestamp, elapsed)
    
    # If recording ended with no eyes detected, add final incident
    if no_eyes_start is not None and eyes_detected_history:
        final_timestamp, _, final_elapsed = eyes_detected_history[-1]
        duration = final_elapsed - no_eyes_start[1]
        if duration > 1:
            no_eyes_detected_incidents.append((no_eyes_start[0], final_timestamp, duration))
    
    # Print emotion summary
    print("\nEMOTION SUMMARY:")
    for emotion, instances in emotion_counts.items():
        avg_confidence = sum(conf for _, conf, _ in instances) / len(instances)
        duration_percentage = (len(instances) / len(emotion_history)) * 100
        print(f"- {emotion.capitalize()}: {len(instances)} instances ({duration_percentage:.1f}% of time), Avg confidence: {avg_confidence:.1f}%")
    
    # Print gaze summary with more detailed direction analysis
    print("\nGAZE SUMMARY:")
    
    # Group similar gaze directions for better analysis
    grouped_gazes = {}
    for gaze, instances in gaze_counts.items():
        # Group by primary direction
        if "left" in gaze:
            key = "Looking left"
        elif "right" in gaze:
            key = "Looking right"
        elif "up" in gaze:
            key = "Looking up"
        elif "down" in gaze:
            key = "Looking down"
        elif "straight" in gaze:
            key = "Looking straight"
        else:
            key = gaze.capitalize()
        
        if key not in grouped_gazes:
            grouped_gazes[key] = []
        grouped_gazes[key].extend(instances)
    
    # Print grouped gaze summary
    for direction, instances in grouped_gazes.items():
        duration_percentage = (len(instances) / len(gaze_history)) * 100
        print(f"- {direction}: {len(instances)} instances ({duration_percentage:.1f}% of time)")
    
    # Print angle statistics
    if angle_history:
        h_angles = [h for _, h, _, _ in angle_history]
        v_angles = [v for _, _, v, _ in angle_history]
        avg_h_angle = sum(h_angles) / len(h_angles)
        avg_v_angle = sum(v_angles) / len(v_angles)
        max_h_angle = max(h_angles)
        min_h_angle = min(h_angles)
        max_v_angle = max(v_angles)
        min_v_angle = min(v_angles)
        
        print("\nGAZE ANGLE STATISTICS:")
        print(f"- Average horizontal angle: {avg_h_angle:.1f}° (Range: {min_h_angle:.1f}° to {max_h_angle:.1f}°)")
        print(f"- Average vertical angle: {avg_v_angle:.1f}° (Range: {min_v_angle:.1f}° to {max_v_angle:.1f}°)")
    
    # Print eye detection summary
    if eyes_detected_history:
        eyes_detected_percentage = (eyes_detected_count / len(eyes_detected_history)) * 100
        print("\nEYE DETECTION SUMMARY:")
        print(f"- Eyes detected: {eyes_detected_count} instances ({eyes_detected_percentage:.1f}% of time)")
        print(f"- Eyes not detected: {len(eyes_detected_history) - eyes_detected_count} instances ({100 - eyes_detected_percentage:.1f}% of time)")
    
    # Print no eyes incidents
    print("\nEYES NOT DETECTED INCIDENTS (>1 second):")
    if no_eyes_detected_incidents:
        for start_time, end_time, duration in no_eyes_detected_incidents:
            print(f"- From {start_time} to {end_time} (Duration: {duration:.1f}s)")
    else:
        print("- No significant incidents where eyes were not detected")
    
    # Print looking away incidents with direction info
    print("\nLOOKING AWAY INCIDENTS (>1 second):")
    if looking_away_incidents:
        for start_time, end_time, duration, gaze_dir in looking_away_incidents:
            print(f"- From {start_time} to {end_time} (Duration: {duration:.1f}s, Direction: {gaze_dir})")
    else:
        print("- No significant looking away incidents detected")
    
    # Print emotional transitions
    print("\nEMOTIONAL TRANSITIONS:")
    prev_emotion = None
    transitions = []
    
    for timestamp, emotion, confidence, elapsed in emotion_history:
        if prev_emotion is not None and emotion != prev_emotion:
            transitions.append((timestamp, prev_emotion, emotion, elapsed))
        prev_emotion = emotion
    
    if transitions:
        for timestamp, from_emotion, to_emotion, elapsed in transitions:
            print(f"- At {timestamp} ({elapsed:.1f}s): {from_emotion} → {to_emotion}")
    else:
        print("- No emotional transitions detected")
    
    # Print overall assessment
    print("\nOVERALL ASSESSMENT:")
    # Determine dominant emotion
    if emotion_counts:
        dominant_emotion = max(emotion_counts.items(), key=lambda x: len(x[1]))[0]
        print(f"- Dominant emotion: {dominant_emotion.capitalize()}")
    else:
        print("- No dominant emotion detected")
    
    # Determine attention level with improved metric
    total_looking_away_time = sum(duration for _, _, duration, _ in looking_away_incidents)
    total_no_eyes_time = sum(duration for _, _, duration in no_eyes_detected_incidents)
    attention_percentage = 100 - ((total_looking_away_time + total_no_eyes_time) / 30) * 100
    if attention_percentage < 0:
        attention_percentage = 0
    elif attention_percentage > 100:
        attention_percentage = 100
        
    print(f"- Attention level: {attention_percentage:.1f}% (looked away for {total_looking_away_time:.1f} seconds, eyes not detected for {total_no_eyes_time:.1f} seconds)")
    
    # Direction preference analysis
    if grouped_gazes:
        preferred_direction = max(grouped_gazes.items(), key=lambda x: len(x[1]))[0]
        if preferred_direction != "Looking straight":
            print(f"- Preferred gaze direction: {preferred_direction}")
    
    # Save report to file
    filename = f"emotion_report_{report_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, 'w') as file:
        file.write(f"EMOTION AND GAZE REPORT #{report_num}\n")
        file.write("="*50 + "\n\n")
        
        file.write("EMOTIONAL TIMELINE:\n")
        for timestamp, emotion, confidence, elapsed in emotion_history:
            file.write(f"{timestamp} ({elapsed:.1f}s): {emotion} ({confidence:.1f}%)\n")
        
        file.write("\nGAZE TIMELINE:\n")
        for timestamp, gaze, elapsed in gaze_history:
            file.write(f"{timestamp} ({elapsed:.1f}s): {gaze}\n")
            
        file.write("\nGAZE ANGLE TIMELINE:\n")
        for timestamp, h_angle, v_angle, elapsed in angle_history:
            file.write(f"{timestamp} ({elapsed:.1f}s): H: {h_angle:.1f}°, V: {v_angle:.1f}°\n")
            
        file.write("\nEYE DETECTION TIMELINE:\n")
        for timestamp, eyes_detected, elapsed in eyes_detected_history:
            file.write(f"{timestamp} ({elapsed:.1f}s): {'Eyes detected' if eyes_detected else 'Eyes NOT detected'}\n")
        
        file.write("\nEMOTION SUMMARY:\n")
        for emotion, instances in emotion_counts.items():
            avg_confidence = sum(conf for _, conf, _ in instances) / len(instances)
            duration_percentage = (len(instances) / len(emotion_history)) * 100
            file.write(f"- {emotion.capitalize()}: {len(instances)} instances ({duration_percentage:.1f}% of time), Avg confidence: {avg_confidence:.1f}%\n")
        
        file.write("\nGAZE SUMMARY:\n")
        for direction, instances in grouped_gazes.items():
            duration_percentage = (len(instances) / len(gaze_history)) * 100
            file.write(f"- {direction}: {len(instances)} instances ({duration_percentage:.1f}% of time)\n")
            
        if angle_history:
            file.write("\nGAZE ANGLE STATISTICS:\n")
            file.write(f"- Average horizontal angle: {avg_h_angle:.1f}° (Range: {min_h_angle:.1f}° to {max_h_angle:.1f}°)\n")
            file.write(f"- Average vertical angle: {avg_v_angle:.1f}° (Range: {min_v_angle:.1f}° to {max_v_angle:.1f}°)\n")
            
        if eyes_detected_history:
            file.write("\nEYE DETECTION SUMMARY:\n")
            file.write(f"- Eyes detected: {eyes_detected_count} instances ({eyes_detected_percentage:.1f}% of time)\n")
            file.write(f"- Eyes not detected: {len(eyes_detected_history) - eyes_detected_count} instances ({100 - eyes_detected_percentage:.1f}% of time)\n")
        
        file.write("\nEYES NOT DETECTED INCIDENTS (>1 second):\n")
        if no_eyes_detected_incidents:
            for start_time, end_time, duration in no_eyes_detected_incidents:
                file.write(f"- From {start_time} to {end_time} (Duration: {duration:.1f}s)\n")
        else:
            file.write("- No significant incidents where eyes were not detected\n")
        
        file.write("\nLOOKING AWAY INCIDENTS (>1 second):\n")
        if looking_away_incidents:
            for start_time, end_time, duration, gaze_dir in looking_away_incidents:
                file.write(f"- From {start_time} to {end_time} (Duration: {duration:.1f}s, Direction: {gaze_dir})\n")
        else:
            file.write("- No significant looking away incidents detected\n")
        
        file.write("\nEMOTIONAL TRANSITIONS:\n")
        if transitions:
            for timestamp, from_emotion, to_emotion, elapsed in transitions:
                file.write(f"- At {timestamp} ({elapsed:.1f}s): {from_emotion} → {to_emotion}\n")
        else:
            file.write("- No emotional transitions detected\n")
        
        file.write("\nOVERALL ASSESSMENT:\n")
        if emotion_counts:
            file.write(f"- Dominant emotion: {dominant_emotion.capitalize()}\n")
        file.write(f"- Attention level: {attention_percentage:.1f}% (looked away for {total_looking_away_time:.1f} seconds, eyes not detected for {total_no_eyes_time:.1f} seconds)\n")
        if grouped_gazes and preferred_direction != "Looking straight":
            file.write(f"- Preferred gaze direction: {preferred_direction}\n")
    
    print(f"\nDetailed report saved to {filename}")
    print("="*50)

def generate_report_thread(emotion_history, gaze_history, angle_history, eyes_detected_history, report_num):
    """Thread function to generate reports without blocking the main program"""
    report_thread = threading.Thread(
        target=generate_emotion_report, 
        args=(emotion_history.copy(), gaze_history.copy(), angle_history.copy(), eyes_detected_history.copy(), report_num)
    )
    report_thread.daemon = True
    report_thread.start()

def load_emotion_icons():
    """Load emotion icons or create placeholders if not available."""
    emotions = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust"]
    emotion_icons = {}
    
    # First try to load from a local directory
    icon_dir = "emotion_icons"
    if os.path.exists(icon_dir):
        for emotion in emotions:
            icon_path = os.path.join(icon_dir, f"{emotion}.png")
            if os.path.exists(icon_path):
                emotion_icons[emotion] = cv2.imread(icon_path)
    
    # If we don't have all icons, create colored placeholders
    for emotion in emotions:
        if emotion not in emotion_icons:
            # Create colored placeholder (100x100 image)
            if emotion == "neutral":
                color = (200, 200, 200)  # Gray
            elif emotion == "happy":
                color = (0, 255, 255)    # Yellow
            elif emotion == "sad":
                color = (255, 0, 0)      # Blue
            elif emotion == "angry":
                color = (0, 0, 255)      # Red
            elif emotion == "surprise":
                color = (255, 0, 255)    # Purple
            elif emotion == "fear":
                color = (255, 255, 0)    # Cyan
            elif emotion == "disgust":
                color = (0, 128, 0)      # Green
            else:
                color = (255, 255, 255)  # White
                
            placeholder = np.ones((100, 100, 3), dtype=np.uint8) * 50  # Dark background
            cv2.rectangle(placeholder, (10, 10), (90, 90), color, -1)  # Colored rectangle
            
            # Draw a face icon based on emotion
            if emotion == "neutral":
                # Neutral face
                cv2.circle(placeholder, (35, 40), 5, (0, 0, 0), -1)  # Left eye
                cv2.circle(placeholder, (65, 40), 5, (0, 0, 0), -1)  # Right eye
                cv2.line(placeholder, (30, 65), (70, 65), (0, 0, 0), 2)  # Straight mouth
            elif emotion == "happy":
                # Happy face
                cv2.circle(placeholder, (35, 40), 5, (0, 0, 0), -1)  # Left eye
                cv2.circle(placeholder, (65, 40), 5, (0, 0, 0), -1)  # Right eye
                cv2.ellipse(placeholder, (50, 60), (20, 10), 0, 0, 180, (0, 0, 0), 2)  # Smile
            elif emotion == "sad":
                # Sad face
                cv2.circle(placeholder, (35, 40), 5, (0, 0, 0), -1)  # Left eye
                cv2.circle(placeholder, (65, 40), 5, (0, 0, 0), -1)  # Right eye
                cv2.ellipse(placeholder, (50, 70), (20, 10), 0, 180, 360, (0, 0, 0), 2)  # Frown
            elif emotion == "angry":
                # Angry face
                cv2.line(placeholder, (25, 35), (40, 40), (0, 0, 0), 2)  # Left eyebrow
                cv2.line(placeholder, (75, 35), (60, 40), (0, 0, 0), 2)  # Right eyebrow
                cv2.circle(placeholder, (35, 45), 5, (0, 0, 0), -1)  # Left eye
                cv2.circle(placeholder, (65, 45), 5, (0, 0, 0), -1)  # Right eye
                cv2.ellipse(placeholder, (50, 70), (20, 10), 0, 180, 360, (0, 0, 0), 2)  # Frown
            elif emotion == "surprise":
                cv2.circle(placeholder, (35, 40), 8, (0, 0, 0), 1)  # Left eye (bigger)
                cv2.circle(placeholder, (65, 40), 8, (0, 0, 0), 1)  # Right eye (bigger)
                cv2.circle(placeholder, (50, 65), 10, (0, 0, 0), 1)  # O-shaped mouth
            elif emotion == "fear":
                # Fearful face
                cv2.circle(placeholder, (35, 40), 7, (0, 0, 0), 1)  # Left eye (wide)
                cv2.circle(placeholder, (65, 40), 7, (0, 0, 0), 1)  # Right eye (wide)
                cv2.ellipse(placeholder, (50, 65), (15, 10), 0, 0, 180, (0, 0, 0), 2)  # Open mouth
                cv2.line(placeholder, (25, 30), (40, 35), (0, 0, 0), 2)  # Left eyebrow raised
                cv2.line(placeholder, (75, 30), (60, 35), (0, 0, 0), 2)  # Right eyebrow raised
            elif emotion == "disgust":
                # Disgusted face
                cv2.circle(placeholder, (35, 40), 5, (0, 0, 0), -1)  # Left eye
                cv2.circle(placeholder, (65, 40), 5, (0, 0, 0), -1)  # Right eye
                cv2.line(placeholder, (30, 65), (70, 55), (0, 0, 0), 2)  # Asymmetric mouth
                cv2.line(placeholder, (25, 35), (40, 40), (0, 0, 0), 2)  # Left eyebrow
            
            emotion_icons[emotion] = placeholder
    
    return emotion_icons

def main():
    # First download/setup the model
    print("Setting up emotion detection model...")
    download_emotion_model()
    # Try to download alternative dataset models
    download_fer_dataset()
    emotion_model = setup_emotion_model()
    
    # Initialize face and eye detection with improved parameters
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Load emotion icons
    emotion_icons = load_emotion_icons()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    cv2.namedWindow("Emotion and Gaze Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Emotion and Gaze Tracking", 800, 600)
    
    # For continuous emotion logging
    emotion_history = []
    gaze_history = []
    angle_history = []
    eyes_detected_history = []
    recording_duration = 30  # seconds
    start_time = time.time()
    continuous_recording = True  # Always recording
    report_counter = 1
    
    # For gaze tracking
    looking_away_start = None
    looking_away_warning = False
    
    print("Starting continuous emotion and gaze tracking. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
            
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # If recording duration has passed, generate report and reset for next interval
        if continuous_recording and elapsed_time >= recording_duration:
            # Generate report in a separate thread to avoid blocking the main program
            generate_report_thread(emotion_history, gaze_history, angle_history, eyes_detected_history, report_counter)
            
            # Reset for next recording period
            start_time = current_time
            emotion_history = []
            gaze_history = []
            angle_history = []
            eyes_detected_history = []
            report_counter += 1
        
        # Detect faces with improved parameters
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))
        
        if len(faces) == 0:
            # No face detected
            if looking_away_start is None:
                looking_away_start = current_time
            
            # Check if looking away for more than 1 second
            if current_time - looking_away_start > 1:
                if not looking_away_warning:
                    print("Warning: No face detected for more than 1 second!")
                    looking_away_warning = True
                
                # Record no face detected event
                if continuous_recording:
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    gaze_history.append((timestamp, "no face detected", elapsed_time))
                    eyes_detected_history.append((timestamp, False, elapsed_time))
            
            # Display message on frame
            cv2.putText(display_frame, "No face detected", 
                      (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
        else:
            looking_away_start = None
            looking_away_warning = False
        
        for (x, y, w, h) in faces:
            # Draw face rectangle (subtle visualization)
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            
            # Detect eyes within face region with improved parameters
            face_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 3, minSize=(30, 30))
            
            # Record eye detection status
            eyes_detected = len(eyes) > 0
            if continuous_recording:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                eyes_detected_history.append((timestamp, eyes_detected, elapsed_time))
            
            # Set default gaze values
            gaze_direction = "unknown"
            h_angle = 0
            v_angle = 0
            
            # Convert eye coordinates to full frame coordinates
            eye_centers = []
            for (ex, ey, ew, eh) in eyes:
                eye_center_x = x + ex + ew//2
                eye_center_y = y + ey + eh//2
                eye_centers.append((eye_center_x, eye_center_y))
                cv2.rectangle(display_frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 0, 255), 1)
            
            # Analyze gaze direction with improved angle-based algorithm
            if len(eye_centers) >= 2:
                gaze_direction, h_angle, v_angle = analyze_eye_position(eye_centers, w, h, x, y)
                
                # Add angle data to history
                if continuous_recording:
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    angle_history.append((timestamp, h_angle, v_angle, elapsed_time))
                
                # Display angle information
                angle_text = f"H: {h_angle:.1f}°, V: {v_angle:.1f}°"
                cv2.putText(display_frame, angle_text, 
                           (x, y-50 if y > 70 else y+h+90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                
                # Check if looking away from center
                if "straight" not in gaze_direction:
                    if looking_away_start is None:
                        looking_away_start = current_time
                    
                    # If looking away for more than 1 second
                    if current_time - looking_away_start > 1:
                        if not looking_away_warning:
                            print(f"Warning: {gaze_direction.capitalize()} for more than 1 second!")
                            looking_away_warning = True
                else:
                    looking_away_start = None
                    looking_away_warning = False
            elif len(eye_centers) == 1:
                gaze_direction = "blinking"
            else:
                gaze_direction = "eyes closed"
                # Add no angle data point
                if continuous_recording:
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    angle_history.append((timestamp, 0, 0, elapsed_time))
            
            # Detect emotion with improved algorithm
            emotion, confidence = detect_emotion(frame, (x, y, w, h), emotion_model)
            
            # Set color based on emotion for visualization
            if emotion == "happy":
                emotion_color = (0, 255, 255)  # Yellow
            elif emotion in ["angry", "disgust", "fear", "sad"]:
                emotion_color = (0, 0, 255)  # Red
            elif emotion == "surprise":
                emotion_color = (255, 0, 255)  # Purple
            else:
                emotion_color = (255, 255, 255)  # White
            
            # Display emotion and gaze information
            emotion_text = f"{emotion} ({confidence:.0f}%)"
            cv2.putText(display_frame, emotion_text, 
                        (x, y-10 if y > 30 else y+h+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2)
            
            gaze_text = f"{gaze_direction}"
            cv2.putText(display_frame, gaze_text, 
                        (x, y-30 if y > 50 else y+h+60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Display emotion icon
            icon_size = 80
            icon_padding = 20
            icon_x = display_frame.shape[1] - icon_size - icon_padding
            icon_y = icon_padding
            
            # Get appropriate emotion icon
            if emotion in emotion_icons:
                emotion_icon = emotion_icons[emotion]
                # Resize icon if needed
                if emotion_icon.shape[0] != icon_size or emotion_icon.shape[1] != icon_size:
                    emotion_icon = cv2.resize(emotion_icon, (icon_size, icon_size))
                
                # Create region of interest
                roi = display_frame[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size]
                
                # Create a mask and apply the icon
                gray_icon = cv2.cvtColor(emotion_icon, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_icon, 10, 255, cv2.THRESH_BINARY)
                icon_fg = cv2.bitwise_and(emotion_icon, emotion_icon, mask=mask)
                icon_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
                dst = cv2.add(icon_bg, icon_fg)
                display_frame[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size] = dst
                
                # Add label below the icon
                label_y = icon_y + icon_size + 20
                cv2.putText(display_frame, emotion.capitalize(), 
                           (icon_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2)
            
            # Record emotion and gaze for continuous recording
            if continuous_recording:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                emotion_history.append((timestamp, emotion, confidence, elapsed_time))
                gaze_history.append((timestamp, gaze_direction, elapsed_time))
        
        # Display countdown to next report
        time_to_next_report = recording_duration - elapsed_time
        report_text = f"Next report in: {time_to_next_report:.0f}s"
        # Place this text in a less obtrusive location (bottom left)
        cv2.putText(display_frame, report_text, 
                    (10, display_frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Display key for angles
        angle_info = "Angles within ±30° = looking straight"
        cv2.putText(display_frame, angle_info, 
                   (10, display_frame.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Emotion and Gaze Tracking", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Generate final report before quitting
            if emotion_history and gaze_history:
                generate_emotion_report(emotion_history, gaze_history, angle_history, eyes_detected_history, report_counter)
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main()