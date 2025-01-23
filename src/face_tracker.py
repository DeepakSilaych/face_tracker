import cv2
import os
from modules.detector import detect_faces
from modules.tracker import initialize_tracker, update_tracker
from modules.utils import draw_bounding_box

def process_video(input_video, output_video):
    """
    Process the input video, detect and track faces, and save the output video.
    
    Args:
        input_video (str): Path to input video file
        output_video (str): Path to save the output video file
    """
    # Initialize video capture
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    trackers = []
    tracking_faces = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Every N frames, detect faces and initialize new trackers
        if len(trackers) == 0:
            face_boxes = detect_faces(frame, gray)
            for bbox in face_boxes:
                tracker = initialize_tracker(frame, bbox)
                trackers.append(tracker)
                tracking_faces.append(bbox)

        # Update trackers
        tracking_faces = []
        trackers_to_remove = []
        
        for i, tracker in enumerate(trackers):
            success, bbox = update_tracker(tracker, frame)
            if success:
                tracking_faces.append(bbox)
                draw_bounding_box(frame, bbox)
            else:
                trackers_to_remove.append(i)

        # Remove failed trackers
        for idx in reversed(trackers_to_remove):
            trackers.pop(idx)

        # Write frame to output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()

if __name__ == "__main__":
    input_video = "data/input_video.mp4"
    output_video = "data/output_video.mp4"
    process_video(input_video, output_video)
