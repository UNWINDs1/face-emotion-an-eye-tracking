#!/usr/bin/env python
import os
import argparse
import shutil
from datetime import datetime
import kagglehub

def download_kaggle_dataset(dataset_name, output_dir):
    """
    Download a dataset from Kaggle using kagglehub
    
    Args:
        dataset_name: Name of the Kaggle dataset (e.g., 'ryersonmultimedialab/ryerson-emotion-database')
        output_dir: Directory to save the downloaded dataset
    """
    print(f"Downloading Kaggle dataset: {dataset_name}")
    
    try:
        # Download dataset using kagglehub
        path = kagglehub.dataset_download(dataset_name)
        print(f"Dataset downloaded to: {path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy files from kagglehub path to our expected data directory
        print(f"Copying files to {output_dir}...")
        if os.path.isdir(path):
            # Get all files recursively
            for root, dirs, files in os.walk(path):
                for file in files:
                    src_path = os.path.join(root, file)
                    # Create a relative path to maintain directory structure
                    rel_path = os.path.relpath(src_path, path)
                    dst_path = os.path.join(output_dir, rel_path)
                    
                    # Ensure destination directory exists
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    
                    # Copy the file
                    shutil.copy2(src_path, dst_path)
        
        print("Dataset downloaded and copied successfully")
        return True
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def process_ravdess_dataset(data_dir, output_dir):
    """
    Process the RAVDESS dataset structure for emotion recognition
    
    Args:
        data_dir: Directory containing the RAVDESS dataset
        output_dir: Directory to organize the dataset
    """
    # Map RAVDESS emotion codes to emotion names
    # RAVDESS Emotion Code: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 
    # 06=fearful, 07=disgust, 08=surprised
    emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy', 
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    # Create the output structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all video files
    video_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.mp4')):
                video_files.append(os.path.join(root, file))
    
    print(f"Found {len(video_files)} video files")
    
    # Create emotion directories
    for emotion in emotion_map.values():
        emotion_dir = os.path.join(output_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
    
    # Organize videos by emotion
    organized_count = 0
    for video_path in video_files:
        filename = os.path.basename(video_path)
        
        # RAVDESS filename format: 
        # Video: 01-01-01-01-01-01-01.mp4
        # Audio: 03-01-01-01-01-01-01.wav
        parts = filename.split('-')
        
        if len(parts) >= 3:
            emotion_code = parts[2]
            if emotion_code in emotion_map:
                emotion = emotion_map[emotion_code]
                
                # Create a directory for the actor if it doesn't exist
                actor_id = parts[6].split('.')[0]
                actor_dir = os.path.join(output_dir, emotion, f"Actor_{actor_id}")
                os.makedirs(actor_dir, exist_ok=True)
                
                # Copy the video file
                dest_path = os.path.join(actor_dir, filename)
                shutil.copy(video_path, dest_path)
                organized_count += 1
    
    print(f"Organized {organized_count} videos by emotion")
    return organized_count > 0

def main():
    parser = argparse.ArgumentParser(description='Train emotion recognition model on Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)')
    
    # Dataset paths
    parser.add_argument('--data_dir', type=str, default='./RAVDESS_Data', 
                        help='Directory to store/find RAVDESS dataset')
    parser.add_argument('--processed_dir', type=str, default='./RAVDESS_Processed', 
                        help='Directory to store organized dataset')
    parser.add_argument('--output_dir', type=str, default='./ravdess_frames', 
                        help='Directory to save extracted frames')
    parser.add_argument('--model_output', type=str, default='./models', 
                        help='Directory to save trained models')
    
    # Dataset download options
    parser.add_argument('--download', action='store_true',
                        help='Download the RAVDESS dataset from Kaggle')
    parser.add_argument('--kaggle_dataset', type=str, 
                        default='ryersonmultimedialab/ryerson-emotion-database',
                        help='Kaggle dataset name')
    
    # Training parameters
    parser.add_argument('--frames_per_video', type=int, default=20,  # Changed from 15 to 20
                        help='Number of frames to extract per video')
    parser.add_argument('--batch_size', type=int, default=64,  # Changed from 32 to 64
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,  # Changed from 20 to 30
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0005, 
                        help='Learning rate')
    parser.add_argument('--recursive', action='store_true', default=True,
                        help='Search subdirectories recursively')
    
    # Operation modes
    parser.add_argument('--skip_frame_extraction', action='store_true',
                        help='Skip frame extraction if frames already exist')
    parser.add_argument('--test_webcam', action='store_true', default=True,  # Changed to default=True
                        help='Test model with webcam after training')
    parser.add_argument('--test_video', type=str, 
                        help='Test model with a specific video file')
    
    # Add option to use existing downloaded data from kagglehub
    parser.add_argument('--use_kagglehub_path', type=str,
                        default=r"C:\Users\Kushagra\.cache\kagglehub\datasets\ryersonmultimedialab\ryerson-emotion-database\versions\1",  # Added your path as default
                        help='Use dataset already downloaded with kagglehub at the specified path')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_output, exist_ok=True)
    
    # Generate timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.model_output, f'ravdess_emotion_model_{timestamp}.pth')
    
    print("=" * 50)
    print(f"RAVDESS Emotion Recognition Training")
    print("=" * 50)
    
    # Use existing kagglehub path if provided
    if args.use_kagglehub_path and os.path.exists(args.use_kagglehub_path):
        print(f"Using dataset downloaded with kagglehub at: {args.use_kagglehub_path}")
        # Copy files from kagglehub path to our data directory
        os.makedirs(args.data_dir, exist_ok=True)
        for root, dirs, files in os.walk(args.use_kagglehub_path):
            for file in files:
                src_path = os.path.join(root, file)
                # Create a relative path to maintain directory structure
                rel_path = os.path.relpath(src_path, args.use_kagglehub_path)
                dst_path = os.path.join(args.data_dir, rel_path)
                
                # Ensure destination directory exists
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                
                # Copy the file
                shutil.copy2(src_path, dst_path)
    # Download dataset if requested
    elif args.download:
        success = download_kaggle_dataset(args.kaggle_dataset, args.data_dir)
        if not success:
            print("Failed to download dataset. Exiting.")
            return
    
    # Check if RAVDESS dataset exists
    if not os.path.exists(args.data_dir):
        print(f"Error: RAVDESS dataset directory not found at {args.data_dir}")
        print("Please use one of these options:")
        print("1. Use the --download flag to download the dataset")
        print("2. Use --use_kagglehub_path to specify a path where you've already downloaded the dataset using kagglehub")
        print("   Example: python script.py --use_kagglehub_path /path/to/kagglehub/download")
        print("3. Use this code to download the dataset and pass the path:")
        print("   ```")
        print("   import kagglehub")
        print("   path = kagglehub.dataset_download('ryersonmultimedialab/ryerson-emotion-database')")
        print("   print('Path to dataset files:', path)")
        print("   ```")
        print("   Then run: python script.py --use_kagglehub_path /path/printed/above")
        return
    
    # Process and organize the dataset
    print(f"\nProcessing RAVDESS dataset...")
    process_ravdess_dataset(args.data_dir, args.processed_dir)
    
    # Import necessary functions from main script
    try:
        from main import prepare_dataset, train_emotion_model, test_emotion_model_webcam, test_emotion_model_video
    except ImportError:
        print("Error: Could not import required functions from main.py.")
        print("Make sure main.py is in the current directory or in the Python path.")
        return
    
    print(f"Dataset directory: {args.processed_dir}")
    print(f"Frames output directory: {args.output_dir}")
    print(f"Model will be saved to: {model_path}")
    print(f"Extracting {args.frames_per_video} frames per video")
    print(f"Training with batch size={args.batch_size}, epochs={args.epochs}, lr={args.learning_rate}")
    print("=" * 50)
    
    # Extract frames and prepare dataset if needed
    if not args.skip_frame_extraction or not os.path.exists(args.output_dir) or len(os.listdir(args.output_dir)) == 0:
        print("\nPreparing dataset and extracting frames...")
        frame_data, emotion_to_idx = prepare_dataset(
            args.processed_dir, 
            args.output_dir,
            args.frames_per_video,
            args.recursive
        )
        
        print(f"\nExtracted frames for {len(frame_data) // args.frames_per_video} videos")
        print(f"Total frames: {len(frame_data)}")
        print("Emotions detected:", list(emotion_to_idx.keys()))
    else:
        print("\nSkipping frame extraction, using existing frames...")
        # Reconstruct frame_data from existing frames
        frame_data = []
        emotion_to_idx = {
            'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3,
            'angry': 4, 'fearful': 5, 'disgust': 6, 'surprised': 7
        }
        
        for emotion in emotion_to_idx.keys():
            emotion_dir = os.path.join(args.output_dir, emotion)
            if os.path.exists(emotion_dir):
                for actor_folder in os.listdir(emotion_dir):
                    actor_path = os.path.join(emotion_dir, actor_folder)
                    if os.path.isdir(actor_path):
                        for frame_file in os.listdir(actor_path):
                            if frame_file.endswith(('.jpg', '.png')):
                                frame_path = os.path.join(actor_path, frame_file)
                                frame_data.append((frame_path, emotion_to_idx[emotion]))
        
        print(f"Found {len(frame_data)} existing frames")
    
    # Train the model if we have frames
    if len(frame_data) > 0:
        print("\nTraining emotion recognition model...")
        trained_model_path = train_emotion_model(
            frame_data,
            emotion_to_idx,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        
        # Copy the model to our versioned path
        import shutil
        shutil.copy(trained_model_path, model_path)
        print(f"Model copied to: {model_path}")
        
        # Test with webcam if requested
        if args.test_webcam:
            print("\nTesting model with webcam...")
            test_emotion_model_webcam(model_path)
        
        # Test with a specific video if provided
        if args.test_video and os.path.exists(args.test_video):
            print(f"\nTesting model with video: {args.test_video}")
            test_emotion_model_video(model_path, args.test_video)
    else:
        print("No frames were found or extracted. Cannot train model.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()