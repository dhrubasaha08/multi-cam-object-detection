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
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

outputFrame = None
lock = threading.Lock()

# Initialize video stream from ESP32-CAM using environment variable
video_stream_url = f'http://{os.getenv("ESP32_CAM_IP")}:81/stream'

def capture_stream():
    global outputFrame, lock

    # Use OpenCV to capture frames from the URL
    cap = cv2.VideoCapture(video_stream_url)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Lock the thread while updating the output frame
        with lock:
            outputFrame = frame.copy()

def generate_frames():
    global outputFrame, lock

    while True:
        # Wait until the frame is available
        with lock:
            if outputFrame is None:
                continue

            # Encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue

        # Yield the output frame in the byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the thread to capture video stream
    t = threading.Thread(target=capture_stream)
    t.daemon = True
    t.start()

    app.run(host='0.0.0.0', port=5000, threaded=True)
