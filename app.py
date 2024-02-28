from flask import Flask, render_template, Response
import cv2
import threading
import numpy as np
from tflite_support.task import vision

app = Flask(__name__)

# Global variables to hold and lock the latest frame
outputFrame = None
lock = threading.Lock()

# Path to your TFLite model file
MODEL_PATH = 'efficientdet_lite0.tflite'

# Initialize the object detector at the start of the application
def initialize_detector():
    base_options = vision.ObjectDetectorOptions(
        file_name=MODEL_PATH, num_threads=4, score_threshold=0.3)
    detector = vision.ObjectDetector.create_from_options(base_options)
    return detector

# Function to capture video stream and apply object detection
def capture_stream(detector):
    global outputFrame, lock

    # Establish connection to the ESP32-CAM video stream
    cap = cv2.VideoCapture('http://10.0.0.224:81/stream')
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert the captured frame to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Prepare the frame for model inference
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        # Perform object detection
        detection_result = detector.detect(input_tensor)

        # Annotate the frame with detection results
        annotated_image = visualize(frame, detection_result)

        # Update the global frame for streaming
        with lock:
            outputFrame = annotated_image.copy()

# Visualize the detection results on the frame
def visualize(image, detection_result):
    for detection in detection_result.detections:
        # Draw bounding boxes and labels on the image
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, (0, 0, 255), 3)

        # Label and score
        category = detection.categories[0]
        category_name = category.category_name
        score = round(category.score, 2)
        label = f"{category_name} ({score})"
        cv2.putText(image, label, (bbox.origin_x, bbox.origin_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image

# Route for serving the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Generator function to stream the output frames
def generate_frames():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode('.jpg', outputFrame)
            if not flag:
                continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Load the object detection model
    detector = initialize_detector()

    # Start the video stream capture on a separate thread
    t = threading.Thread(target=capture_stream, args=(detector,))
    t.daemon = True
    t.start()

    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, threaded=True)
