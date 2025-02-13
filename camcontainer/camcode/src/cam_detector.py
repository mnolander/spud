import queue
import numpy as np
import cv2
import threading
import logging
import sys
import os
import tf.transformations as tf_transforms  # ✅ For converting rotation matrix to quaternion

# Dynamically add the enhanced_python_aprilgrid/src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "enhanced_python_aprilgrid", "src")))

# Import AprilTag Detector
from aprilgrid import Detector
from stereo_processor import StereoProcessor
from frame_processing import frame_to_gray_np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cv2.setNumThreads(4)  # Use multi-threading for better performance

class DetectorThread(threading.Thread):
    """
    A worker thread that:
      - Pulls frames from the detection_queue
      - Runs AprilTag detection
      - Stores results in detection_result_queue
      - Computes 3D positions and rotations if both cameras detect the same tags
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

        # Define AprilTag 3D Model Points (16 cm tag)
        self.tag_size = 0.16  # 16 cm
        self.tag_corners_3d = np.array([
            [-self.tag_size / 2, -self.tag_size / 2, 0],  
            [self.tag_size / 2, -self.tag_size / 2, 0],  
            [self.tag_size / 2, self.tag_size / 2, 0],  
            [-self.tag_size / 2, self.tag_size / 2, 0]  
        ], dtype=np.float32)

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
        """Compute 3D positions and rotations when both cameras detect the same tags."""
        left_detections = self.latest_detections["DEV_1AB22C00E123"]
        right_detections = self.latest_detections["DEV_1AB22C00E588"]

        if not left_detections or not right_detections:
            return  # No detections from one of the cameras

        matched_ids = set(left_detections.keys()).intersection(set(right_detections.keys()))
        if not matched_ids:
            return  # No common detections

        left_pts = np.array([left_detections[tag_id] for tag_id in matched_ids])
        right_pts = np.array([right_detections[tag_id] for tag_id in matched_ids])

        # ✅ Compute 3D positions
        points_4d_hom = cv2.triangulatePoints(
            self.stereo_processor.P1,
            self.stereo_processor.P2,
            left_pts.T,
            right_pts.T
        )
        points_3d = (points_4d_hom[:3] / points_4d_hom[3]).T  # Convert to Euclidean

        results = []
        for tag_id, (x, y, z) in zip(matched_ids, points_3d):
            print(f"Tag ID {tag_id}: 3D Position (X: {x:.6f}, Y: {y:.6f}, Z: {z:.6f})")

            # ✅ Estimate rotation using solvePnP
            if tag_id in left_detections:
                corners_2d = np.array(left_detections[tag_id], dtype=np.float32)
                camera_matrix = self.stereo_processor.K1
                dist_coeffs = self.stereo_processor.calibration.primary_distortion
            else:
                corners_2d = np.array(right_detections[tag_id], dtype=np.float32)
                camera_matrix = self.stereo_processor.K2
                dist_coeffs = self.stereo_processor.calibration.secondary_distortion

            success, rvec, _ = cv2.solvePnP(self.tag_corners_3d, corners_2d, camera_matrix, dist_coeffs)
            if success:
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                quaternion = tf_transforms.quaternion_from_matrix(
                    np.vstack([np.hstack([rotation_matrix, [[0], [0], [0]]]), [0, 0, 0, 1]])
                )
            else:
                quaternion = [0.0, 0.0, 0.0, 1.0]  # Default if pose estimation fails

            results.append((tag_id, (x, y, z), quaternion))

        return results

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

            # Compute 3D positions and rotations
            tag_results = self.compute_3d_positions()

            # Push rectified frame & detections with rotation to frame_consumer.py
            self.detection_result_queue.put((cam_id, gray_full, detections_fullres, tag_results))
