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

api_base_url = "http://192.168.3.21:5002"
CACHE_DIR = Path(__file__).parent.parent / "cache" / "analysis"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model():
    model_path = Path(__file__).parent.parent / "models" / "yolo11m-pose.pt"
    
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

    # def extreme_test(self):
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
