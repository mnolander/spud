import queue
import cv2
import threading
import logging
import time
from cam_detector import DetectorThread
from constants import FRAME_QUEUE_SIZE, NUM_DETECTOR_THREADS, DETECTION_FRAME_SKIP
from vmbpy import *
from frame_processing import resize_for_display
from utils import create_dummy_frame
from stereo_processor import StereoProcessor
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimize OpenCV for speed
cv2.setUseOptimized(True)
cv2.setNumThreads(4)  # Use multi-threading for better performance

# ------------- FrameConsumer -------------
class FrameConsumer:
    """
    Main consumer class that:
      1) Pulls frames from the frame_queue.
      2) Hands them off to the parallel detector threads (detection_queue),
         optionally skipping frames to reduce CPU load.
      3) Receives detection results (detection_result_queue).
      4) Draws detection overlays and displays in a window.
    """

    def __init__(self, frame_queue: queue.Queue, num_detector_threads: int = 4):
        self.frame_queue = frame_queue

        # Queues for handing off frames to the detector threads
        self.detection_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.detection_result_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)

        # Create multiple parallel detector threads
        self.detector_threads = []
        for i in range(num_detector_threads):
            dt = DetectorThread(self.detection_queue, self.detection_result_queue)
            dt.start()
            self.detector_threads.append(dt)

        self.stereo_processor = StereoProcessor()

        # For tracking FPS, etc.
        self.camera_data = {}

        # Latest frames and detection results by camera ID
        self.last_frames = {}
        self.latest_results = {}

        # --- MODIFIED: We’ll track how many frames we’ve handled per camera for skipping ---
        self.frame_counts = {}

    def log_frame_info(self, cam_id: str, frame: 'Frame'):
        """Log frame resolution."""
        resolution = (frame.get_width(), frame.get_height())
        print(f"Camera {cam_id} - Resolution: {resolution}")

    def run(self):
        """Main loop: fetch frames, pass to detector, fetch results, draw and display."""
        IMAGE_CAPTION = 'DroneCam Multicam View: Press <Enter> to exit'
        KEY_CODE_ENTER = 13

        alive = True
        cv2.namedWindow(IMAGE_CAPTION, cv2.WINDOW_AUTOSIZE)

        while alive:
            # 1. Retrieve new frames from producers
            try:
                cam_id, frame = self.frame_queue.get(timeout=0.01)
                if frame:
                    # Valid frame
                    self.log_frame_info(cam_id, frame)
                    self.last_frames[cam_id] = frame

                    # --- MODIFIED: Counting frames per camera to skip if needed ---
                    if cam_id not in self.frame_counts:
                        self.frame_counts[cam_id] = 0
                    self.frame_counts[cam_id] += 1

                    # Only offload to detector on every Nth frame to reduce CPU load
                    if self.frame_counts[cam_id] % DETECTION_FRAME_SKIP == 0:
                        try:
                            self.detection_queue.put_nowait((cam_id, frame))
                        except queue.Full:
                            pass
                else:
                    # Frame is None => camera missing or ended
                    self.last_frames.pop(cam_id, None)
                    self.latest_results.pop(cam_id, None)
                    try:
                        self.detection_queue.put_nowait((cam_id, None))
                    except queue.Full:
                        pass

            except queue.Empty:
                # No new frames at the moment
                pass

            # 2. Collect detection results
            while True:
                try:
                    cam_id_res, gray_res, detections_res = self.detection_result_queue.get_nowait()
                    if gray_res is None or detections_res is None:
                        # If detection result is None, the camera ended
                        self.latest_results.pop(cam_id_res, None)
                    else:
                        # Store the latest detection results
                        self.latest_results[cam_id_res] = (gray_res, detections_res)
                except queue.Empty:
                    break

            # 3. Display results
            if self.latest_results:
                processed_frames = []
                # Sort camera IDs for a consistent layout
                for cid in sorted(self.latest_results.keys()):
                    gray, detections = self.latest_results[cid]

                    # Convert to BGR to draw color overlays
                    color_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                    # Draw AprilTag corners and IDs
                    for det in detections:
                        tag_id = det['tag_id']
                        corners = det['corners']  # Already scaled to full res

                        # Draw corners in different colors
                        for i, corner in enumerate(corners):
                            corner = corner.flatten().astype(int)
                            if i == 0:
                                color = (0, 0, 255)      # Red
                            elif i == 1:
                                color = (0, 255, 0)      # Green
                            elif i == 2:
                                color = (255, 0, 0)      # Blue
                            else:
                                color = (0, 255, 255)    # Yellow
                            cv2.circle(color_frame, tuple(corner), 8, color, -1)

                        # Draw lines between corners
                        for i in range(4):
                            start = tuple(corners[i].flatten().astype(int))
                            end = tuple(corners[(i + 1) % 4].flatten().astype(int))
                            cv2.line(color_frame, start, end, (255, 255, 255), 2)

                        # Put ID text at the center
                        center = np.mean(corners, axis=0).flatten().astype(int)
                        cv2.putText(color_frame, str(tag_id), tuple(center),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # Resize for display
                    display_frame = resize_for_display(color_frame)
                    processed_frames.append(display_frame)

                # Concatenate side by side
                if processed_frames:
                    display_frame = np.concatenate(processed_frames, axis=1)
                    cv2.imshow(IMAGE_CAPTION, display_frame)
            else:
                # If no detections or no cameras, show dummy frame
                cv2.imshow(IMAGE_CAPTION, create_dummy_frame())

            # 4. Check for user exit
            if KEY_CODE_ENTER == cv2.waitKey(1):
                cv2.destroyAllWindows()
                alive = False

        # Stop the parallel detector threads gracefully
        for dt in self.detector_threads:
            dt.stop()
        for dt in self.detector_threads:
            dt.join()
