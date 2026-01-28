from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from utils.processing import Processing
import threading
from uuid import uuid4
import os
import traceback
import json

proc = Processing()
app = Flask(__name__, static_folder='static')

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def run_flask_server():
    app.run(host="0.0.0.0", port=3000, debug=False)

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
