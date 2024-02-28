from flask import Flask, render_template, Response
import cv2
import threading

app = Flask(__name__)

# Global variable to hold the latest frame captured from the ESP32-CAM
outputFrame = None
# Lock to ensure thread-safe exchanges of the output frame
lock = threading.Lock()

# URL of the ESP32-CAM video stream
ESP32_CAM_STREAM_URL = 'http://10.0.0.224:81/stream'

def capture_stream():
    """
    Connects to the ESP32-CAM and captures the video stream.
    The latest frame is stored in a global variable.
    """
    global outputFrame, lock

    # Create a VideoCapture object to read from the ESP32-CAM stream
    cap = cv2.VideoCapture(ESP32_CAM_STREAM_URL)

    while True:
        # Read the next frame from the stream
        ret, frame = cap.read()
        if not ret:
            continue

        # Update the global outputFrame under lock protection
        with lock:
            outputFrame = frame.copy()

def generate_frames():
    """
    Encodes the latest frame as JPEG and yields it as a byte stream,
    following the multipart/x-mixed-replace content type which is necessary for streaming.
    """
    global outputFrame, lock

    while True:
        # Wait for the next frame available
        with lock:
            if outputFrame is None:
                continue

            # Encode the frame in JPEG format
            success, encodedImage = cv2.imencode('.jpg', outputFrame)
            if not success:
                continue

        # Yield the frame data
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/')
def index():
    """
    Renders the main page with the video stream embedded.
    """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    Route to serve the video feed.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start a thread to capture the video stream
    t = threading.Thread(target=capture_stream)
    t.daemon = True
    t.start()

    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, threaded=True)
