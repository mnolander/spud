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
from stereo_processor import StereoProcessor
from frame_processing import frame_to_gray_np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# cv2.setUseOptimized(True)
cv2.setNumThreads(4)  # Use multi-threading for better performance

class DetectorThread(threading.Thread):
    """
    A worker thread that:
      - Pulls frames from the detection_queue
      - Runs AprilTag detection
      - Stores results in detection_result_queue
      - Computes 3D positions if both cameras detect the same tags
      - Pushes rectified images to frame_consumer.py
    """

    def __init__(self, detection_queue: queue.Queue, detection_result_queue: queue.Queue):
        super().__init__()
        self.detection_queue = detection_queue
        self.detection_result_queue = detection_result_queue
        self.detector = Detector('t16h5b1')
        self.killswitch = threading.Event()
        self.stereo_processor = StereoProcessor()

        # Stores latest detections from both cameras
        self.latest_detections = {"DEV_1AB22C00E123": {}, "DEV_1AB22C00E588": {}}

    def stop(self):
        self.killswitch.set()

    def rectify_frame(self, cam_id, frame):
        """Applies stereo rectification based on the camera ID."""
        if cam_id == "DEV_1AB22C00E123":
            return cv2.remap(frame, self.stereo_processor.map1x, self.stereo_processor.map1y, interpolation=cv2.INTER_LINEAR)
        elif cam_id == "DEV_1AB22C00E588":
            return cv2.remap(frame, self.stereo_processor.map2x, self.stereo_processor.map2y, interpolation=cv2.INTER_LINEAR)
        return frame  # Return original if unknown cam_id

    def compute_3d_positions(self):
        """Compute 3D positions when both cameras detect the same tags."""
        left_detections = self.latest_detections["DEV_1AB22C00E123"]
        right_detections = self.latest_detections["DEV_1AB22C00E588"]

        if not left_detections or not right_detections:
            return  # No detections from one of the cameras

        matched_ids = set(left_detections.keys()).intersection(set(right_detections.keys()))

        if not matched_ids:
            return  # No common detections

        left_pts = np.array([left_detections[tag_id] for tag_id in matched_ids])
        right_pts = np.array([right_detections[tag_id] for tag_id in matched_ids])

        points_4d_hom = cv2.triangulatePoints(
            self.stereo_processor.P1,
            self.stereo_processor.P2,
            left_pts.T,
            right_pts.T
        )
        points_3d = (points_4d_hom[:3] / points_4d_hom[3]).T

        for tag_id, (x, y, z) in zip(matched_ids, points_3d):
            print(f"Tag ID {tag_id}: 3D Position (X: {x:.6f}, Y: {y:.6f}, Z: {z:.6f})")

    def run(self):
        while not self.killswitch.is_set():
            try:
                cam_id, frame = self.detection_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            if frame is None:
                logger.warning(f"Detector received an empty frame from {cam_id}")
                self.detection_result_queue.put((cam_id, None, None, None))
                continue

            logger.info(f"Processing frame from {cam_id}")

            # Convert to grayscale
            gray_full = frame_to_gray_np(frame)

            # Validate frame
            if gray_full is None or gray_full.size == 0:
                logger.error(f"Invalid frame from {cam_id}, skipping detection")
                continue

            # Apply rectification
            rectified_frame = self.rectify_frame(cam_id, gray_full)

            if rectified_frame is None or rectified_frame.size == 0:
                logger.error(f"Rectified frame is empty from {cam_id}")
                continue

            # Perform AprilTag detection
            detections = self.detector.detect(rectified_frame)
            detections_fullres = []

            # Store detections
            detected_tags = {}
            for det in detections:
                center = np.array(det.corners).mean(axis=0)  # Use center for 3D matching
                detected_tags[det.tag_id] = center
                detections_fullres.append({'tag_id': det.tag_id, 'corners': np.array(det.corners)})

            # Save detections for 3D computation
            self.latest_detections[cam_id] = detected_tags

            # Compute 3D positions when both cameras detect tags
            self.compute_3d_positions()

            # Push rectified frame & detections to frame_consumer.py
            self.detection_result_queue.put((cam_id, gray_full, detections_fullres))
