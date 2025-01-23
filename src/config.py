import os
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEO_CONFIG = {
    'input_path': os.path.join(BASE_DIR, 'data', 'input_video.mp4'),
    'output_path': os.path.join(BASE_DIR, 'data', 'output_video.mp4'),
    'frame_width': 640,
    'frame_height': 480,
    'fps': 30
}

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
DETECTION_CONFIG = {
    'scaleFactor': 1.1,
    'minNeighbors': 5,
    'minSize': (30, 30)
}
