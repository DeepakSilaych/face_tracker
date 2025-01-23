import cv2
from config import VIDEO_CONFIG, FACE_CASCADE, DETECTION_CONFIG

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=DETECTION_CONFIG['scaleFactor'],
        minNeighbors=DETECTION_CONFIG['minNeighbors'],
        minSize=DETECTION_CONFIG['minSize']
    )
    return faces

def process_video():
    cap = cv2.VideoCapture(VIDEO_CONFIG['input_path'])
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()
