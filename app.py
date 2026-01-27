from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from utils.processing import Processing
import threading


proc = Processing()
app = Flask(__name__, static_folder='static')

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
    return jsonify({"status": "Upload endpoint hit"})

if __name__ == '__main__':
    # Run the Flask server on a separate thread

    # Start the thread
    flask_thread = threading.Thread(target=run_flask_server)
    flask_thread.start()
