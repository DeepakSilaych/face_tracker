import cv2

def draw_bounding_box(frame, bbox, color=(0, 255, 0), thickness=2):
    """
    Draws a rectangle around the detected object.

    Args:
        frame (ndarray): Video frame.
        bbox (tuple): Bounding box (x, y, w, h).
        color (tuple): Rectangle color (default is green).
        thickness (int): Line thickness.
    """
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    return frame
