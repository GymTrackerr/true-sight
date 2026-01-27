from ultralytics import YOLO
from utils.joints import Joints, frame_save
from utils.exercisedb import ExerciseDBSearch
import torch
from pathlib import Path

api_base_url = "http://192.168.3.21:5002"

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
        """Find exercise in ExerciseDB"""
        found = self.edb.search(exercise_name)
        if (not found):
            raise ValueError(f"Exercise '{exercise_name}' not found in ExerciseDB")

        cur0 = self.get_results(api_base_url+found.get('images')[0])
        cur1 = self.get_results(api_base_url+found.get('images')[1])

        change = cur0[0].find_changing_angle(cur1[0])
        print(f"Change between two frames: {change[0]} with diff {change[1]}")

    def get_results(self, imagePath):
        results = self.model(source=imagePath, save=True, save_txt=True)
        # Access the results, one per person in frame
        curs = []

        for result in results:
            curs.append(frame_save(result))
            # result.save(filename="result.jpg")  # save to disk
        return curs
