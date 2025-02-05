import cv2

# Frame Configuration
FRAME_QUEUE_SIZE = 10
FRAME_HEIGHT = 3036
FRAME_WIDTH = 4024
DISPLAY_WIDTH = 720
DISPLAY_HEIGHT = 720
NUM_DETECTOR_THREADS = 6

# Detection Configuration
DETECTION_FRAME_SKIP = 2  # Detect on every Nth frame
DETECTION_DOWNSCALE = 2   # Downscale factor for detection

# OpenCV optimizations
cv2.setUseOptimized(True)
