import numpy as np


def extract_corner_and_id_list_apriltags(tags):
    april_corners = [np.asarray(tag.corners, dtype=np.float32).reshape(1, 4, 2) for tag in tags]
    april_ids = np.asarray([tag.tag_id for tag in tags])
    return april_corners, april_ids


def draw_markers(image, corners, ids):
    return cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids, (0, 255, 0))
