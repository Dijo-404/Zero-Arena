from flask import Flask, render_template, request, jsonify, send_file
import cv2
from ultralytics import YOLO
import numpy as np
import os
from werkzeug.utils import secure_filename
import io
from PIL import Image
import base64

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# --- YOLO Model Setup ---
model_loaded = False
model = None

def initialize_resources():
    global model, model_loaded
    # Load the YOLOv8 model
    try:
        model = YOLO('foduucom/shelf-object-detection-yolov8')
        model_loaded = True
        print("YOLO model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}. Running in no-model mode.")
        # Create a dummy model object if loading fails
        class DummyModel:
            def __call__(self, frame):
                class DummyResult:
                    def __init__(self, original_frame):
                        self.orig_img = original_frame
                    def plot(self):
                        return self.orig_img
                return [DummyResult(frame)]
        model = DummyModel()
        model_loaded = False

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_error_image(message, width=640, height=480):
    """Creates a black image with an error message."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Calculate text size and position for centering
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(frame, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

def process_image(image_path):
    """Process image with YOLO model and return result."""
    try:
        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            return create_error_image("Error: Could not read image file")

        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        return annotated_frame

    except Exception as e:
        print(f"Error processing image: {e}")
        return create_error_image(f"Error processing image: {str(e)}")

@app.route('/')
def index():
    """Image upload home page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and processing."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP, TIFF'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add timestamp to avoid filename conflicts
            import time
            timestamp = str(int(time.time()))
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{timestamp}{ext}"

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the image
            processed_frame = process_image(filepath)

            # Save the processed image
            result_filename = f"result_{filename}"
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            cv2.imwrite(result_path, processed_frame)

            # Convert processed image to base64 for display
            _, buffer = cv2.imencode('.jpg', processed_frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            return jsonify({
                'success': True,
                'image': img_base64,
                'filename': result_filename,
                'model_status': 'loaded' if model_loaded else 'dummy'
            })

    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed image."""
    try:
        result_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
        if os.path.exists(result_path):
            return send_file(result_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Download error: {str(e)}'}), 500

@app.route('/status')
def status():
    """Get model and app status."""
    return jsonify({
        'model_loaded': model_loaded,
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'results_folder': app.config['RESULTS_FOLDER'],
        'allowed_extensions': list(ALLOWED_EXTENSIONS)
    })

# Cleanup old files (optional)
@app.route('/cleanup')
def cleanup_files():
    """Clean up old uploaded and result files."""
    try:
        import time
        current_time = time.time()
        files_removed = 0

        # Remove files older than 1 hour (3600 seconds)
        for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > 3600:  # 1 hour
                        os.remove(file_path)
                        files_removed += 1

        return jsonify({
            'success': True,
            'files_removed': files_removed,
            'message': 'Cleanup completed'
        })

    except Exception as e:
        return jsonify({'error': f'Cleanup error: {str(e)}'}), 500

        # Remove files older than 1 hour (3600 seconds)
        for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > 3600:  # 1 hour
                        os.remove(file_path)
                        files_removed += 1

        return jsonify({
            'success': True,
            'files_removed': files_removed,
            'message': 'Cleanup completed'
        })

    except Exception as e:
        return jsonify({'error': f'Cleanup error: {str(e)}'}), 500

if __name__ == '__main__':
    initialize_resources()
    app.run(debug=True, threaded=True, use_reloader=False)

# Cleanup function for app termination
def cleanup_on_exit():
    """Clean up resources on app exit."""
    print("Application shutting down...")
    # Add any cleanup code here if needed

import atexit
atexit.register(cleanup_on_exit)
