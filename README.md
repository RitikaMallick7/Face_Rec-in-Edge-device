# Face Recognition in Edge Devices

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Edge Computing](https://img.shields.io/badge/Edge_Computing-Optimized-green.svg)
![Framework](https://img.shields.io/badge/Framework-OpenCV%20%7C%20DeepLearning-orange.svg)

This repository focuses on deploying efficient face recognition models on edge devices (like Raspberry Pi, Jetson Nano, or local IoT gateways). The goal is to provide low-latency, privacy-focused facial identification without relying heavily on cloud processing.

##  Features

* **Optimized Performance:** Lightweight models designed for hardware with limited computational resources.
* **Real-time Detection:** Fast face detection and matching.
* **Local Storage:** Face embeddings are stored locally to ensure data privacy.
* **Extensible Architecture:** Easily integrate new face datasets and models.

##  Tech Stack

* **Language:** Python
* **Libraries:** OpenCV, NumPy, Dlib / Face_Recognition
* **Deployment:** Edge Devices (ARM-based architectures)

##  Installation

```bash
# Clone the repository
git clone [https://github.com/asutoshp10/Face_Rec-in-Edge-device.git](https://github.com/asutoshp10/Face_Rec-in-Edge-device.git)

# Navigate to the directory
cd Face_Rec-in-Edge-device
```

## Roadmap & Future Scope
We are actively working on expanding the capabilities of this system. The next major update will include:

RTSP Protocol Integration: We will be moving beyond local webcams to support RTSP (Real Time Streaming Protocol). This will allow the edge device to ingest live video feeds from IP cameras and network streams for remote surveillance and monitoring.

Multi-Face Tracking: Improving the ability to track multiple individuals simultaneously in a single frame.

Hardware Acceleration: Leveraging TensorRT or OpenVINO for faster inference on supported edge hardware.

# Install dependencies
```bash
pip install -r requirements.txt
```
