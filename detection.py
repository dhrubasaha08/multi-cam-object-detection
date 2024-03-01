'''
Copyright 2024 Dhruba Saha <dhrubasaha@outlook.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

'''

from flask import Flask, render_template, Response
import cv2
import threading
from dotenv import load_dotenv
import numpy as np
from tflite_support.task import vision
from tflite_support.task import processor
from tflite_support.task import core

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

outputFrame = None
lock = threading.Lock()

MODEL_PATH = 'efficientdet_lite0.tflite'
video_stream_url = f'http://{os.getenv("ESP32_CAM_IP1")}:81/stream'
_TEXT_COLOR = (0, 0, 255)

def initialize_detector():
    # Initialize the object detection model
    base_options = core.BaseOptions(file_name=MODEL_PATH, num_threads=4, use_coral=False)
    detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)
    return detector

def capture_stream(detector):
    global outputFrame, lock

    cap = cv2.VideoCapture(video_stream_url)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert the frame to the RGB colorspace
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Prepare the image for detection
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        # Perform object detection
        detection_result = detector.detect(input_tensor)

        # Visualize the results
        annotated_image = visualize(frame, detection_result)

        # Update the global frame
        with lock:
            outputFrame = annotated_image.copy()

def visualize(image, detection_result):
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

        category = detection.categories[0]
        label = f"{category.category_name} ({round(category.score, 2)})"
        cv2.putText(image, label, (bbox.origin_x, bbox.origin_y + 20), cv2.FONT_HERSHEY_PLAIN, 1, _TEXT_COLOR, 1)

    return image

def generate_frames():
    global outputFrame, lock

    while True:
        with lock:
            if outputFrame is None:
                continue

            # Encode the frame in JPEG format
            success, encodedImage = cv2.imencode('.jpg', outputFrame)
            if not success:
                continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    detector = initialize_detector()
    t = threading.Thread(target=capture_stream, args=(detector,))
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
