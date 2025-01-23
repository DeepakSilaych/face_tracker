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
