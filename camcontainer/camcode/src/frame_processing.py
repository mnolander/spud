import cv2
import numpy as np
from vmbpy import *

from constants import DISPLAY_WIDTH, DISPLAY_HEIGHT, DETECTION_DOWNSCALE

def resize_for_display(frame: np.ndarray) -> np.ndarray:
    """Downscale for display in an OpenCV window."""
    return cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)

def frame_to_gray_np(frame):
    """
    Convert a frame to grayscale NumPy array. 
    If the frame is already a NumPy array, return it directly.
    """
    if isinstance(frame, np.ndarray):  
        return frame 
    return frame.as_opencv_image()


def downscale_for_detection(gray_full: np.ndarray) -> np.ndarray:
    """Downscale image for detection to reduce CPU load."""
    if DETECTION_DOWNSCALE <= 1:
        return gray_full
    h, w = gray_full.shape[:2]
    return cv2.resize(gray_full, (w // DETECTION_DOWNSCALE, h // DETECTION_DOWNSCALE), interpolation=cv2.INTER_LINEAR)

def upscale_corners(corners, scale_factor_x, scale_factor_y):
    """
    Scale AprilTag corner coordinates back to the full-sized image if detection was done on
    a downscaled version.
    """
    return [corner * np.array([scale_factor_x, scale_factor_y]) for corner in corners]
