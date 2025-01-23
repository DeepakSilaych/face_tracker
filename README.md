This is a face tracker that processes a video file instead of a live camera feed. The implementation is structured to ensure readability and scalability.

---

### **Modular File Structure**

```
face-tracker/
├── src/
│   ├── face_tracker.py       # Main script to run the tracker
│   ├── modules/
│   │   ├── detector.py       # Face detection logic
│   │   ├── tracker.py        # Tracking logic
│   │   ├── utils.py          # Utility functions (e.g., draw bounding boxes)
├── data/
│   ├── input_video.mp4       # Input video for testing
│   ├── output_video.mp4      # Processed output video with tracking
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
```

---

### **Implementation**

#### **Step 1: Create `detector.py`**
This module handles face detection using `dlib`.

```python
import dlib

def detect_faces(frame, gray_frame):
    """
    Detects faces in the given frame using dlib's face detector.
    
    Args:
        frame (ndarray): Original color frame.
        gray_frame (ndarray): Grayscale version of the frame.

    Returns:
        list: A list of bounding boxes [(x, y, w, h), ...].
    """
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray_frame)
    bboxes = [(face.left(), face.top(), face.width(), face.height()) for face in faces]
    return bboxes
```

---

#### **Step 2: Create `tracker.py`**
This module manages the tracking of detected faces using OpenCV’s trackers.

```python
import cv2

def initialize_tracker(frame, bbox):
    """
    Initializes a tracker for a given bounding box.

    Args:
        frame (ndarray): Initial frame where the object is located.
        bbox (tuple): Bounding box (x, y, w, h).

    Returns:
        cv2.Tracker: Initialized tracker object.
    """
    tracker = cv2.TrackerKCF_create()  # You can replace with other trackers
    tracker.init(frame, bbox)
    return tracker

def update_tracker(tracker, frame):
    """
    Updates the tracker for the current frame.

    Args:
        tracker (cv2.Tracker): Initialized tracker object.
        frame (ndarray): Current video frame.

    Returns:
        tuple: Success flag and updated bounding box (success, bbox).
    """
    return tracker.update(frame)
```

---

#### **Step 3: Create `utils.py`**
Utility functions for drawing bounding boxes and video processing.

```python
import cv2

def draw_bounding_box(frame, bbox, color=(0, 255, 0)):
    """
    Draws a rectangle around the detected object.

    Args:
        frame (ndarray): Video frame.
        bbox (tuple): Bounding box (x, y, w, h).
        color (tuple): Rectangle color (default is green).
    """
    x, y, w, h = map(int, bbox)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

def save_video(output_path, frame_width, frame_height, fps):
    """
    Initializes the video writer to save the processed video.

    Args:
        output_path (str): Path to save the output video.
        frame_width (int): Width of the video frames.
        frame_height (int): Height of the video frames.
        fps (int): Frames per second of the video.

    Returns:
        cv2.VideoWriter: Video writer object.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
```

---

#### **Step 4: Create `face_tracker.py`**
The main script to handle video processing and integrate the modules.

```python
import cv2
import os
from modules.detector import detect_faces
from modules.tracker import initialize_tracker, update_tracker
from modules.utils import draw_bounding_box, save_video

def main(input_video, output_video):
    # Check if input video exists
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video {input_video} not found.")

    # Open the input video
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    writer = save_video(output_video, frame_width, frame_height, fps)

    tracking = False
    tracker = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not tracking:
            # Detect faces in the current frame
            bboxes = detect_faces(frame, gray_frame)
            if bboxes:
                # Initialize tracker for the first detected face
                tracker = initialize_tracker(frame, bboxes[0])
                tracking = True
        else:
            # Update tracker
            success, bbox = update_tracker(tracker, frame)
            if success:
                draw_bounding_box(frame, bbox)
            else:
                tracking = False  # Stop tracking if it fails

        # Write the frame to the output video
        writer.write(frame)

        # Display the processed frame
        cv2.imshow("Face Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video = "data/input_video.mp4"
    output_video = "data/output_video.mp4"
    main(input_video, output_video)
```

---

### **Execution Steps**

1. **Place Input Video**:
   Save your test video as `data/input_video.mp4`.

2. **Install Dependencies**:
   Add these dependencies to `requirements.txt`:
   ```text
   opencv-python
   dlib
   imutils
   numpy
   ```
   Install them:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Script**:
   Execute the face tracker:
   ```bash
   python src/face_tracker.py
   ```

4. **Output**:
   - Processed video with bounding boxes is saved as `data/output_video.mp4`.

---

Let me know if you need help refining this implementation or adding additional features!