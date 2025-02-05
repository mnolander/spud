import queue
import numpy as np
import cv2
import threading
import logging
import sys
import os

# Dynamically add the enhanced_python_aprilgrid/src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "enhanced_python_aprilgrid", "src")))

# Import AprilTag Detector
from aprilgrid import Detector

from frame_processing import downscale_for_detection, frame_to_gray_np
from utils import *
from frame_processing import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cv2.setUseOptimized(True)  # Enables OpenCV optimization
cv2.setNumThreads(4)  # Use multiple CPU threads for processing


class DetectorThread(threading.Thread):
    """
    A worker thread that:
      - Pulls frames from the detection_queue
      - Runs AprilTag detection
      - Stores results in detection_result_queue
    """

    def __init__(self, detection_queue: queue.Queue, detection_result_queue: queue.Queue):
        super().__init__()
        self.detection_queue = detection_queue
        self.detection_result_queue = detection_result_queue
        self.detector = Detector('t16h5b1')
        self.killswitch = threading.Event()

        # Camera intrinsics for distance calculation
        self.camera_intrinsics = {
            "DEV_1AB22C00E123": [7061.970133146324, 7055.239263382366, 2129.3033141629267, 1627.5503251130253],
            "DEV_1AB22C00E588": [7074.560676188498, 7062.053568655379, 2037.2857092571062, 1549.725255281428]
        }

        # Camera distortion coefficients
        self.camera_dis = {
            "DEV_1AB22C00E123": [-0.29254, 0.82733, -0.0019, -7.17e-05],
            "DEV_1AB22C00E588": [-0.27067, 0.69110, -0.00084, -0.00051]
        }

        # Physical size of the AprilTag (meters)
        self.tag_size = 0.018  

    def stop(self):
        self.killswitch.set()

    def run(self):
        while not self.killswitch.is_set():
            try:
                cam_id, frame = self.detection_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            if frame is None:
                logger.warning(f"Detector received an empty frame from {cam_id}")
                self.detection_result_queue.put((cam_id, None, None))
                continue

            logger.info(f"Processing frame from {cam_id}")

            # Convert to grayscale full resolution
            gray_full = frame_to_gray_np(frame)

            # Validate frame
            if not isinstance(gray_full, np.ndarray) or gray_full.size == 0:
                logger.error(f"Invalid frame from {cam_id}, skipping detection")
                continue

            # Downscale for detection
            # gray_small = downscale_for_detection(gray_full)
            gray_small = gray_full

            if gray_small is None or gray_small.size == 0:
                logger.error(f"Downscaled frame is empty from {cam_id}")
                continue

            # Perform detection
            detections_small = self.detector.detect(gray_small)

            # Check if detections were found
            if not detections_small:
                logger.warning(f"No detections found in {cam_id}")
            else:
                logger.info(f"Detected {len(detections_small)} AprilTags in {cam_id}")

            # Scale corners back to full resolution
            scale_x = gray_full.shape[1] / gray_small.shape[1]
            scale_y = gray_full.shape[0] / gray_small.shape[0]
            detections_fullres = []

            for det in detections_small:
                scaled_corners = upscale_corners(det.corners, scale_x, scale_y)

                # Calculate distance if camera intrinsics are available
                if cam_id in self.camera_intrinsics:
                    print("dection gotten")
                    # distance = compute_distance(self.camera_intrinsics[cam_id], self.tag_size, scaled_corners)
                    # pos = compute_pose(
                    #     self.camera_intrinsics[cam_id],
                    #     self.tag_size,
                    #     np.array([corner[0] for corner in scaled_corners], dtype=np.float32),
                    #     self.camera_dis[cam_id]
                    # )

                    # # Log detected tag position
                    # if pos:
                    #     logger.info(f"Camera {cam_id} - Tag {det.tag_id}: {pos['pose']['position']}")

                    # if distance is not None:
                    #     logger.info(f"Camera {cam_id} - Tag {det.tag_id}: Distance = {distance:.2f} meters")
                    # else:
                    #     logger.warning(f"Camera {cam_id} - Tag {det.tag_id}: Invalid corner structure")

                # Append detection result
                detections_fullres.append({
                    'tag_id': det.tag_id,
                    'corners': scaled_corners
                })

            # Push results back
            self.detection_result_queue.put((cam_id, gray_full, detections_fullres))
