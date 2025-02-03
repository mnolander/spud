import queue
import numpy as np
import cv2
import threading
# -----------------------------------------------------------------------------
# --- ADDED FOR APRILTAG DETECTION ---
import sys
import os

# Dynamically add the enhanced_python_aprilgrid/src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "enhanced_python_aprilgrid", "src")))

# Now, you can import the Detector class
from aprilgrid import Detector

import numpy as np
# -----------------------------------------------------------------------------

from frame_processing import downscale_for_detection, frame_to_gray_np
from utils import *
from frame_processing import *

class DetectorThread(threading.Thread):
    """
    A single worker thread that pulls frames from a shared detection_queue,
    runs AprilTag detection on a downscaled image (if enabled),
    and puts results into a shared detection_result_queue.
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

        # cam dis_coefficient
        self.camera_dis = {
            "DEV_1AB22C00E123": [-0.29254001960563103, 0.8273391822433487, -0.0019010343025743829, -7.179588729872542e-05],
            "DEV_1AB22C00E588": [-0.2706765105538779, 0.6911079417360364, -0.0008490143333884144, -0.0005189366405546061]
        }

        # Physical size of the AprilTag (in meters)
        self.tag_size = 0.018  # Example: 16.2 cm

    def stop(self):
        self.killswitch.set()

    def run(self):
        while not self.killswitch.is_set():
            try:
                # Get a (cam_id, frame) pair from the detection queue
                cam_id, frame = self.detection_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            if frame is None:
                # Camera or stream ended
                self.detection_result_queue.put((cam_id, None, None))
                continue

            # Convert to grayscale full resolution
            gray_full = frame_to_gray_np(frame)

            # Downscale for detection
            gray_small = downscale_for_detection(gray_full)

            # Perform detection at lower res
            detections_small = self.detector.detect(gray_small)

            # Scale corners back to full resolution
            scale_x = gray_full.shape[1] / gray_small.shape[1]
            scale_y = gray_full.shape[0] / gray_small.shape[0]
            detections_fullres = []

            for det in detections_small:
                scaled_corners = upscale_corners(det.corners, scale_x, scale_y)

                # Calculate distance if camera intrinsics are available
                if cam_id in self.camera_intrinsics:

                    distance = compute_distance(self.camera_intrinsics[cam_id], self.tag_size, scaled_corners)
                    pos = compute_pose(
                        self.camera_intrinsics[cam_id],
                        self.tag_size,
                        np.array([corner[0] for corner in scaled_corners], dtype=np.float32),
                        self.camera_dis[cam_id]
                    )
                    print(pos)
                    
                    # Save the data to a JSON file
                    # with open("pose_data.json", "w") as json_file:
                    #     json.dump(pos, json_file, indent=4)  # `indent=4` makes the JSON file readable
                    if distance is not None:
                        print(f"Camera {cam_id} - Tag ID {det.tag_id}: Distance = {distance:.2f} meters")
                    else: # Invalid corner structure
                        print(f"Camera {cam_id} - Tag ID {det.tag_id}: Invalid corner structure")

                # Append detection result
                detections_fullres.append({
                    'tag_id': det.tag_id,
                    'corners': scaled_corners
                })

            # Push results back
            self.detection_result_queue.put((cam_id, gray_full, detections_fullres))