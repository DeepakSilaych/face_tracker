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
    tracker = cv2.TrackerKCF_create()  # Using KCF tracker for better performance
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
