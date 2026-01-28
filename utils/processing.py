from ultralytics import YOLO
from utils.joints import Joints, frame_save
from utils.exercisedb import ExerciseDBSearch
import torch
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import json
import cv2
import os

api_base_url = "http://192.168.3.21:5002"
CACHE_DIR = Path(__file__).parent.parent / "cache" / "analysis"
OUTPUT_DIR = Path(__file__).parent.parent / "static" / "output"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model():
    model_path = Path(__file__).parent.parent / "models" / "yolo26n-pose.pt"
    
    # Create models directory if it doesn't exist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If model doesn't exist, YOLO will auto-download it
    if not model_path.exists():
        print(f"Model not found at {model_path}, downloading...")
        model = YOLO("yolo11m-pose")  # YOLO auto-downloads from hub
        # Save to models directory
        model.export(format='pt', save_dir=str(model_path.parent))
    else:
        print(f"Loading model from {model_path}")
        model = YOLO(str(model_path))
    
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(device))
        model.to(device)
    return model

class Processing:
    """Processing images, comparing to start and end pose images from ExerciseDB"""
    edb = ExerciseDBSearch()
    model = load_model()

    def __init__(self):
        pass

    def process_video(self, video_path, exercise_name, exercise_data, draw_skeleton, output_dir=None):
        """
        Process video with YOLO pose detection and track joint changes throughout.
        
        Args:
            video_path: Path to input video
            exercise_name: Name of exercise being performed
            exercise_data: Reference exercise data from ExerciseDB
            draw_skeleton: Boolean to draw skeleton on output video
            output_dir: Directory to save output video (default: output/)
        
        Returns:
            output_path: Path to saved processed video
        """
        if output_dir is None:
            output_dir = OUTPUT_DIR
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video to get properties
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Generate output filename (just UUID.mp4)
        input_filename = Path(video_path).stem
        output_filename = f"{input_filename}.mp4"
        output_path = output_dir / output_filename
        
        # Setup video writer for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Process video frame by frame
        frame_joints_array = []  # Store all frame joints
        frame_comparisons = []   # Store comparison data between consecutive frames
        frame_count = 0
        
        print(f"Processing video: {exercise_name}")
        print(f"Total frames: {total_frames}")

        results = self.model(video_path, save=True)

        while True:
            ret, frame = cap.read()
            print(f"Processing frame {frame_count+1}/{total_frames}")
            if not ret:
                break
            
            # Run YOLO pose detection on frame
            results = self.model(frame)
            
            # Save frame joints
            frame_data = []
            for result in results:
                frame_data.append(frame_save(result))
            
            frame_joints_array.append(frame_data)
            
            # Compare with previous frame if exists
            if len(frame_joints_array) >= 2:
                prev_frame = frame_joints_array[-2]
                curr_frame = frame_joints_array[-1]
                
                if len(prev_frame) > 0 and len(curr_frame) > 0:
                    # Compare first person in frame (primary subject) - only if both frames have detections
                    if prev_frame[0].has_detections() and curr_frame[0].has_detections():
                        changes = prev_frame[0].find_changing_joints(curr_frame[0])
                        if changes:  # Only add if there are actual changes
                            frame_comparisons.append({
                                'frame': frame_count,
                                'changes': changes
                            })
            
            # Draw skeleton if requested
            if draw_skeleton == 'true' and len(results) > 0:
                # Plot the pose skeleton on the frame
                output_frame = results[0].plot()  # Returns numpy array (BGR)
            else:
                output_frame = frame
            
            # Write frame to output video
            out.write(output_frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"\nAnalysis complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Total frame comparisons: {len(frame_comparisons)}")
        print(f"Video saved to: {output_path}")
        
        # Create completion marker JSON file
        status_file = output_dir / f"{output_path.stem}.json"
        try:
            with open(status_file, 'w') as f:
                json.dump({"status": "complete"}, f)
            print(f"Completion marker created: {status_file}")
        except IOError as e:
            print(f"Warning: Could not create status file: {e}")
        
        # Return the output path
        return str(output_path)

    def find_exercise(self, exercise_name: str):
        """Find exercise in ExerciseDB with caching"""
        # Create cache directory if it doesn't exist
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create cache file path using exercise name
        cache_file = CACHE_DIR / f"{exercise_name.replace(' ', '_').lower()}.json"
        
        # Check if results are cached
        if cache_file.exists():
            try:
                print(f"Loading cached results for '{exercise_name}'")
                with open(cache_file, 'r') as f:
                    cached_changes = json.load(f)
                return cached_changes
            except (json.JSONDecodeError, IOError) as e:
                print(f"Cache read failed: {e}, recomputing...")
        
        # Not in cache, compute results
        found = self.edb.search(exercise_name)
        if (not found):
            raise ValueError(f"Exercise '{exercise_name}' not found in ExerciseDB")

        cur0 = self.get_results(api_base_url+found.get('images')[0])
        cur1 = self.get_results(api_base_url+found.get('images')[1])

        changes = cur0[0].find_changing_joints(cur1[0])
        print(f"Analysis complete: {changes[0]} with diff {changes[1]}")
        
        # Save results to cache
        try:
            with open(cache_file, 'w') as f:
                json.dump(changes, f, indent=2)
            print(f"Cached results to {cache_file}")
        except IOError as e:
            print(f"Warning: Could not save cache: {e}")
        
        return changes

    def get_results(self, imagePath):
        # Download image to memory (BytesIO) instead of disk
        response = requests.get(imagePath, timeout=10)
        response.raise_for_status()
        
        # Load image from bytes into memory
        img = Image.open(BytesIO(response.content))
        img_array = np.array(img)
        
        # Pass image array to model (stays in RAM, no disk save)
        results = self.model(source=img_array, save=False)
        
        # Access the results, one per person in frame
        curs = []

        for result in results:
            curs.append(frame_save(result))
        return curs
