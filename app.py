from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from utils.processing import Processing
import threading
from uuid import uuid4
import os
import traceback
import json
import cv2
import tempfile
import shutil

proc = Processing()
app = Flask(__name__, static_folder='static')

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def run_flask_server():
    app.run(host="0.0.0.0", port=3000, debug=False)

def transcode_video(input_path, target_fps=15, target_width=640):
    """Transcode video to lower framerate and resolution"""
    try:
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if fps == 0 or width == 0 or height == 0:
            print(f"Warning: Could not read video properties, skipping transcode")
            cap.release()
            return
        
        # Calculate aspect ratio and new height
        aspect_ratio = height / width
        new_width = target_width
        new_height = int(target_width * aspect_ratio)
        
        # Write to temporary file first
        temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
        os.close(temp_fd)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, target_fps, (new_width, new_height))
        
        if not out.isOpened():
            print(f"Warning: Could not open video writer, skipping transcode")
            cap.release()
            os.remove(temp_path)
            return
        
        frame_count = 0
        frame_skip = max(1, round(fps / target_fps))  # Skip frames to reduce framerate (round up for more aggressive skipping)
        
        print(f"Transcoding video: {fps}fps -> {target_fps}fps, {width}x{height} -> {new_width}x{new_height}, skipping every {frame_skip} frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames to reduce framerate
            if frame_count % frame_skip == 0:
                resized = cv2.resize(frame, (new_width, new_height))
                out.write(resized)
            
            frame_count += 1
        
        cap.release()
        out.release()
        
        frames_written = frame_count // frame_skip
        print(f"Video transcoding complete: {frames_written} frames written")
        
        # Replace original file with transcoded version
        if frames_written > 0:
            shutil.move(temp_path, input_path)
            print(f"Original video replaced with transcoded version")
        else:
            print(f"Warning: No frames written, keeping original video")
            os.remove(temp_path)
        
    except Exception as e:
        print(f"Warning: Video transcode failed: {str(e)}")
        # Continue anyway - original video will be used
        try:
            cap.release()
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        proc.find_exercise("Barbell Curl")
        # Handle form submission or other POST requests here
        return jsonify({"status": "POST request received"})
    
@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        exercise_name = data.get('exercise_name')
        
        if not exercise_name:
            return jsonify({"status": "error", "message": "Exercise name required"}), 400
        
        # Get analysis result from processing
        result = proc.find_exercise(exercise_name)
        
        return jsonify({
            "status": "success", 
            "exercise_name": exercise_name,
            "analysis": result
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload():
    try:
        exercise_name = request.form.get('exercise')
        webcam = request.form.get('webcamstream', 'false')
        draw_skeleton = request.form.get('drawSkeleton', 'false')
        
        if not exercise_name:
            return jsonify({"status": "error", "message": "Exercise name required"}), 400
        
        # Get exercise data for reference
        exercise_data = proc.find_exercise(exercise_name)
        
        if webcam == 'false':
            if 'video' not in request.files:
                return jsonify({"status": "error", "message": "No video file provided"}), 400
            
            file = request.files['video']
            
            if file.filename == '':
                return jsonify({"status": "error", "message": "No file selected"}), 400
            
            if file:
                filename = uuid4().hex + ".mp4"
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(video_path)
                
                # Transcode video to lower framerate (15fps) and resolution (640p)
                transcode_video(video_path, target_fps=15, target_width=640)
                
                # Start processing in background thread
                process_thread = threading.Thread(
                    target=proc.process_video,
                    args=(video_path, exercise_name, exercise_data),
                    kwargs={"draw_skeleton": draw_skeleton}
                )
                process_thread.daemon = True
                process_thread.start()
                
                # Return UUID immediately, frontend will poll for status
                return jsonify({
                    "status": "success",
                    "message": "Video uploaded and processing started",
                    "request_id": filename,
                    "exercise_name": exercise_name
                }), 200
        
        return jsonify({"status": "error", "message": "Webcam support not implemented yet"}), 501
        
    except Exception as e:
        print(f"ERROR in /upload: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/status/<path:request_id>', methods=['GET'])
def check_status(request_id):
    """Check if video processing is complete"""
    try:
        # Check if completion marker JSON file exists
        output_dir = os.path.join(app.root_path, 'static', 'output')
        uuid_name = os.path.splitext(request_id)[0]  # Remove .mp4
        status_file = os.path.join(output_dir, f"{uuid_name}.json")
        
        if os.path.exists(status_file):
            # Verify the status file contains completion status
            try:
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                    if status_data.get("status") == "complete":
                        relative_path = f"/static/output/{request_id}"
                        return jsonify({
                            "status": "complete",
                            "output_path": relative_path
                        }), 200
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Error reading status file: {e}")
        
        # Still processing
        return jsonify({
            "status": "processing",
            "message": "Video is still being processed"
        }), 202
        
    except Exception as e:
        print(f"ERROR in /api/status: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    # Run the Flask server on a separate thread

    # Start the thread
    flask_thread = threading.Thread(target=run_flask_server)
    flask_thread.start()
