from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)
# --- YOLO Model and Camera Setup ---
model_loaded = False
model = None
camera = None

def initialize_resources():
    global model, model_loaded, camera
    # Load the YOLOv8 model
    try:
        model = YOLO('/home/dijo404/git/Zero-Arena/yolov8s.pt')
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
    # Initialize camera
    camera = cv2.VideoCapture(0)  # Use 0 for web camera
    if not camera.isOpened():
        print("Error: Could not open camera.")
        camera = None
def create_error_frame(message):
    """Creates a black frame with an error message."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, message, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()
def gen_frames():
    """Video streaming generator function."""
    if camera is None:
        error_message = "Camera not found."
        if not model_loaded:
            error_message = "YOLO model and camera not found."
        frame_bytes = create_error_frame(error_message)
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            frame_bytes = create_error_frame("Error reading frame.")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            break
        else:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    initialize_resources()
    app.run(debug=True, threaded=True, use_reloader=False)
# When the app is terminated, release the camera
@app.teardown_appcontext
def teardown_appcontext(exception=None):
    global camera
    if camera and camera.isOpened():
        camera.release()
        print("Camera released.")

