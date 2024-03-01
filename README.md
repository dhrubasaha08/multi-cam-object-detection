# Multi-Camera Object Detection System

This repository hosts the implementation necessary to establish a multi-camera object detection system leveraging the power of ESP32-CAMs and a Raspberry Pi. The system captures video streams from ESP32-CAM modules and applies object detection using TensorFlow Lite, demonstrating a foundational approach to integrating edge devices with AI capabilities.

## Getting Started

### Prerequisites

Ensure the Raspberry Pi is connected to the internet and have terminal access. The setup assumes it's a fresh installation of Raspberry Pi OS and the ESP32-CAM modules ready and accessible within the same network.

### Raspberry Pi Setup

1. **System Update and Dependencies Installation**

   updating and installing necessary packages:

   ```bash
   sudo apt update && sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
   ```

2. **Python 3.9 Installation**

   Compile and install Python 3.9 on Raspberry Pi to ensure compatibility with TensorFlow Lite and other dependencies:

   ```bash
   wget https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tar.xz
   tar xf Python-3.9.0.tar.xz
   cd Python-3.9.0
   ./configure --enable-optimizations --prefix=/usr
   make -j $(nproc)  # Utilizes all available cores for compilation
   sudo make altinstall
   ```

   Verify the installation:

   ```bash
   python3.9 --version
   ```

3. **Virtual Environment Preparation**

   Setting up a Python virtual environment isolates the project dependencies and avoids conflicts with system packages:

   ```bash
   python3.9 -m venv tflite
   source tflite/bin/activate
   ```

4. **Environment Configuration**

   Properly configure the environment to connect to the ESP32-CAM modules:

   ```bash
   echo "ESP32_CAM_IP=<ESP32-CAM IP Address>" > .env
   ```

   Replace `<ESP32-CAM IP Address>` with the actual IP address of ESP32-CAM.

5. **Application Execution**

   - **Basic Video Streaming:**

     To verify the basic video stream functionality from the ESP32-CAM, execute:

     ```bash
     python3 basic.py
     ```

   - **Object Detection:**

     To run the object detection on the video stream:

     ```bash
     python3 detection.py
     ```

## License

This project is distributed under the Apache License 2.0, which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly cited.