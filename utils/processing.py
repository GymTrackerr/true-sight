from ultralytics import YOLO
from utils.joints import Joints, ConnectedJoints, frame_save
from utils.templates import build_template_from_export, save_template
from utils.exercisedb import ExerciseDBSearch
from utils.analyzer import LiveWindowAnalyzer
from utils.overlay import draw_pose_and_metrics

import torch
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import json
import cv2
import os

CACHE_DIR = Path(__file__).parent.parent / "cache" / "analysis"
OUTPUT_DIR = Path(__file__).parent.parent / "static" / "output"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ScaledFrameSave:
    """Wrapper around frame_save that scales joint coordinates from low-res to full-res space"""
    def __init__(self, frame_obj, scale_x, scale_y):
        self.frame_obj = frame_obj
        self.scale_x = scale_x
        self.scale_y = scale_y
    
    def has_detections(self):
        return self.frame_obj.has_detections()
    
    def get_joint_xy(self, joint: Joints):
        """Get joint coordinates scaled to full resolution"""
        pt = self.frame_obj.get_joint_xy(joint)
        if pt is None:
            return None
        return (pt[0] * self.scale_x, pt[1] * self.scale_y)
    
    def get_connected_joints(self, connected: ConnectedJoints):
        return (
            self.get_joint_xy(connected.value[1][0]),
            self.get_joint_xy(connected.value[1][1]),
            self.get_joint_xy(connected.value[1][2]),
        )
    
    def midpoint(self, j1: Joints, j2: Joints):
        """Calculate midpoint between two joints (scaled)"""
        loc1 = self.get_joint_xy(j1)
        loc2 = self.get_joint_xy(j2)
        if loc1 is None or loc2 is None:
            return None
        return ((loc1[0] + loc2[0]) / 2, (loc1[1] + loc2[1]) / 2)
    
    def shoulder_width(self):
        """Calculate shoulder width"""
        return self.find_width(Joints.LEFT_SHOULDER, Joints.RIGHT_SHOULDER)
    
    def hip_width(self):
        """Calculate hip width"""
        return self.find_width(Joints.LEFT_HIP, Joints.RIGHT_HIP)
    
    def torso_center(self):
        sm = self.midpoint(Joints.LEFT_SHOULDER, Joints.RIGHT_SHOULDER)
        hm = self.midpoint(Joints.LEFT_HIP, Joints.RIGHT_HIP)
        if sm is None or hm is None:
            return None
        return ((sm[0] + hm[0]) / 2.0, (sm[1] + hm[1]) / 2.0)
    
    def find_width(self, j1: Joints, j2: Joints):
        """Calculate width between two joints (scaled)"""
        loc1 = self.get_joint_xy(j1)
        loc2 = self.get_joint_xy(j2)
        if loc1 is None or loc2 is None:
            return None
        dx = loc1[0] - loc2[0]
        dy = loc1[1] - loc2[1]
        return (dx**2 + dy**2)**0.5
    
    def _dist(self, j1: Joints, j2: Joints):
        a = self.get_joint_xy(j1)
        b = self.get_joint_xy(j2)
        if a is None or b is None:
            return None
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx*dx + dy*dy) ** 0.5
    
    def scale(self):
        """Get scale factor (use from original for analysis consistency)"""
        return self.frame_obj.scale()
    
    def angles_dict(self):
        """Get angles dict (use from original, angles don't change with scale)"""
        return self.frame_obj.angles_dict()
    
    def segment_lengths_norm(self):
        """Get normalized segment lengths (use from original, normalized values don't change)"""
        return self.frame_obj.segment_lengths_norm()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model():
    model_path = Path(__file__).parent.parent / "models" / "yolo26s-pose.pt"
    
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

    def process_video(self, video_path, exercise_name, exercise_data, draw_skeleton, output_dir=None, target_fps=15, target_width=640):
        """
        Process video with YOLO pose detection and track joint changes throughout.
        
        Args:
            video_path: Path to input video
            exercise_name: Name of exercise being performed
            exercise_data: Reference exercise data from ExerciseDB
            draw_skeleton: Boolean to draw skeleton on output video
            output_dir: Directory to save output video (default: output/)
            target_fps: Target FPS for processing (default: 15)
            target_width: Target width for processing (default: 640)
        
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
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Original video: {orig_width}x{orig_height} @ {orig_fps} fps")
        
        # Calculate processing resolution (maintain aspect ratio)
        aspect_ratio = orig_height / orig_width
        proc_width = target_width
        proc_height = int(target_width * aspect_ratio)
        
        # Calculate frame skip to match target fps
        frame_skip = max(1, round(orig_fps / target_fps))
        output_fps = orig_fps / frame_skip
        
        print(f"Processing: {proc_width}x{proc_height}, skipping {frame_skip} frames ({output_fps:.1f} fps output)")
        
        # Calculate scale factors
        scale_x = orig_width / proc_width
        scale_y = orig_height / proc_height
        
        # Generate output filename
        input_filename = Path(video_path).stem
        output_filename = f"{input_filename}.mp4"
        output_path = output_dir / output_filename
        
        # Setup video writer for output (original resolution, reduced fps)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (orig_width, orig_height))
        
        if not out.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")
        
        print(f"Output: {orig_width}x{orig_height} @ {output_fps:.1f} fps -> {output_path}")
        
        # Process video frame by frame
        frame_joints_array = []  # Store all frame joints
        frame_comparisons = []   # Store comparison data between consecutive frames
        frame_count = 0
        
        print(f"Processing video: {exercise_name}")
        print(f"Total frames: {orig_total_frames}")

        # results = self.model(video_path, save=True)
        exercise = self.find_exercise(exercise_name)
        
        # Load template for scoring
        from utils.templates import load_template
        from utils.overlay import draw_rep_score
        
        template = None
        template_path = CACHE_DIR / f"{exercise_name.replace(' ', '_').lower()}_template.json"
        if template_path.exists():
            try:
                template = load_template(str(template_path))
                print(f"Loaded template: {exercise_name}")
            except Exception as e:
                print(f"Warning: Could not load template: {e}")

        analyzer = LiveWindowAnalyzer(fps=output_fps, window_seconds=1.0, template=template)
        live_out = []
        m = None
        last_rep_score = None
        
        frame_count = 0
        output_frame_count = 0
        
        while True:
            ret, frame_orig = cap.read()
            
            if not ret:
                break
            
            # Skip frames to reduce fps
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            output_frame_count += 1
            
            # Resize for processing
            frame_proc = cv2.resize(frame_orig, (proc_width, proc_height))
            
            # Run YOLO pose detection
            results = self.model(frame_proc)
            
            # Save frame joints (scale coordinates back to original resolution)
            frame_data = []
            for result in results:
                frame_obj = frame_save(result)
                frame_data.append(ScaledFrameSave(frame_obj, scale_x, scale_y))
            
            frame_joints_array.append(frame_data)
            
            # should only be one person in frame, so take first result
            ## TODO: Handle multiple people in frame in future


            # Default: no data this frame
            angles = {}
            torso_norm = None
            seglens = None

            if len(frame_data) > 0 and frame_data[0].has_detections():
                curr = frame_data[0]

                angles = curr.angles_dict()

                tc = curr.torso_center()
                sc = curr.scale()
                torso_norm = (tc[0]/sc, tc[1]/sc) if (tc and sc) else None

                seglens = curr.segment_lengths_norm()

                m = analyzer.update(angles=angles, torso_center_norm=torso_norm, seglens_norm=seglens)

            if m:
                live_out.append(m.to_dict())
                # Track rep score if available
                if m.rep_score:
                    last_rep_score = m.rep_score




            # # Compare with previous frame if exists
            # if len(frame_joints_array) >= 2:
            #     prev_frame = frame_joints_array[-2]
            #     curr_frame = frame_joints_array[-1]
                
            #     if len(prev_frame) > 0 and len(curr_frame) > 0:
            #         # Compare first person in frame (primary subject) - only if both frames have detections
            #         if prev_frame[0].has_detections() and curr_frame[0].has_detections():
            #             changes = prev_frame[0].find_changing_joints(curr_frame[0])
            #             if changes:  # Only add if there are actual changes
            #                 frame_comparisons.append({
            #                     'frame': frame_count,
            #                     'changes': changes
            #                 })
            
            # Draw skeleton if requested
            # if 
                # Decide what to write out
            output_frame = frame_orig

            if draw_skeleton == 'true':
                # Draw using our overlay (works even if YOLO has no detections)
                metrics_dict = m.to_dict() if m else None
                person0 = frame_data[0] if (len(frame_data) > 0 and frame_data[0].has_detections()) else None
                output_frame = draw_pose_and_metrics(output_frame, person0, metrics_dict)
                
                # Draw previous rep's score on right side
                output_frame = draw_rep_score(output_frame, last_rep_score)

            # Write frame to output video
            out.write(output_frame)
            frame_count += 1
            
            if output_frame_count % 30 == 0:
                print(f"Processed {output_frame_count} output frames")
        
        # Release resources
        cap.release()
        out.release()

        with open(output_dir / f"{input_filename}_live_metrics.json", "w") as f:
            json.dump(live_out, f, indent=2)

        print(f"\nAnalysis complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Total frame comparisons: {len(frame_comparisons)}")
        json.dump(frame_comparisons, open(output_dir / f"{input_filename}_comparisons.json", 'w'), indent=2)
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

        images = self.edb.get_image_urls(found)
        if len(images) < 2:
            raise ValueError(f"Exercise '{exercise_name}' does not have enough reference images for comparison")
        
        cur0 = self.get_results(images[0])
        cur1 = self.get_results(images[1])

        changes = cur0[0].find_changing_joints(cur1[0])
        print(f"Analysis complete: {changes[0]} with diff {changes[1]}")

        tpl = build_template_from_export(exercise_name, changes)
        save_template(str(CACHE_DIR / f"{exercise_name.replace(' ', '_').lower()}_template.json"), tpl)

        return tpl.to_dict()  # or return tpl directly
        # Save results to cache
        # try:
        #     with open(cache_file, 'w') as f:
        #         json.dump(changes, f, indent=2)
        #     print(f"Cached results to {cache_file}")
        # except IOError as e:
        #     print(f"Warning: Could not save cache: {e}")
        
        # return changes

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
