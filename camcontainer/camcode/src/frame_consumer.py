import queue
import cv2
import threading
import logging
import time
from detector import DetectorThread
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

class FrameConsumer:
    """
    Main consumer class that:
      1) Displays **a single combined window** for both camera feeds.
      2) Runs stereo rectification **in a background thread**.
      3) Sends rectified frames for detection **without blocking display**.
    """

    def __init__(self, frame_queue: queue.Queue, num_detector_threads: int = 8):  # Increased detector threads
        self.frame_queue = frame_queue
        self.detection_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.detection_result_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)

        # Create detector threads
        self.detector_threads = [DetectorThread(self.detection_queue, self.detection_result_queue) for _ in range(num_detector_threads)]
        for dt in self.detector_threads:
            dt.start()

        self.stereo_processor = StereoProcessor()

        self.last_frames = {}
        self.latest_results = {}
        self.frame_counts = {}
        self.logged_resolutions = set()

        # Track processing times for debugging
        self.last_update_time = time.time()

    def log_frame_info(self, cam_id: str, frame: 'Frame'):
        """Log frame resolution only once per camera."""
        resolution = (frame.get_width(), frame.get_height())
        if cam_id not in self.logged_resolutions:
            logger.info(f"Camera {cam_id} - Resolution: {resolution}")
            self.logged_resolutions.add(cam_id)

    def run(self):
        IMAGE_CAPTION = 'DroneCam Multicam View: Press <Enter> to exit'
        KEY_CODE_ENTER = 13
        cv2.namedWindow(IMAGE_CAPTION, cv2.WINDOW_AUTOSIZE)

        while True:
            try:
                cam_id, frame = self.frame_queue.get(timeout=0.01)
                if frame:
                    self.log_frame_info(cam_id, frame)
                    opencv_image = frame.as_opencv_image()
                    self.last_frames[cam_id] = opencv_image

                    # **Display raw frame immediately**
                    self.display_combined(IMAGE_CAPTION)

                    # **Process only every Nth frame**
                    self.frame_counts[cam_id] = self.frame_counts.get(cam_id, 0) + 1
                    if self.frame_counts[cam_id] % DETECTION_FRAME_SKIP == 0:

                        # Ensure stereo rectification only happens when two frames are available
                        if len(self.last_frames) >= 2:
                            cam_ids = list(self.last_frames.keys())
                            image_cam0 = self.last_frames[cam_ids[0]]
                            image_cam1 = self.last_frames[cam_ids[1]]

                            # **Run rectification in background**
                            threading.Thread(target=self.rectify_and_detect, args=(image_cam0, image_cam1, cam_ids[0], cam_ids[1]), daemon=True).start()

            except queue.Empty:
                pass  # No new frames

            # Show **one** combined display
            self.display_combined(IMAGE_CAPTION)

            # Exit on Enter key
            if KEY_CODE_ENTER == cv2.waitKey(1):
                cv2.destroyAllWindows()
                break

        # Stop detector threads
        for dt in self.detector_threads:
            dt.stop()
        for dt in self.detector_threads:
            dt.join()

    def rectify_and_detect(self, image_cam0, image_cam1, cam_id0, cam_id1):
        """Runs stereo rectification and sends frames for detection."""
    
        # Default values
        rectified_cam0 = None
        rectified_cam1 = None

        self.stereo_processor.rectify_frames_async(image_cam0, image_cam1, cam_id0, cam_id1)

        time.sleep(0.05)  # Small delay to allow async processing

        # Retrieve rectified images
        with self.stereo_processor.lock:
            if cam_id0 in self.stereo_processor.rectified_frames:
                rectified_cam0 = self.stereo_processor.rectified_frames.pop(cam_id0)
            if cam_id1 in self.stereo_processor.rectified_frames:
                rectified_cam1 = self.stereo_processor.rectified_frames.pop(cam_id1)

        # Ensure valid images
        if rectified_cam0 is None or rectified_cam1 is None:
            logger.error(f"Rectified images are None for {cam_id0} or {cam_id1}!")
            return

        # Convert to grayscale **only if needed**
        gray_cam0 = rectified_cam0 if len(rectified_cam0.shape) == 2 else cv2.cvtColor(rectified_cam0, cv2.COLOR_BGR2GRAY)
        gray_cam1 = rectified_cam1 if len(rectified_cam1.shape) == 2 else cv2.cvtColor(rectified_cam1, cv2.COLOR_BGR2GRAY)

        # Remove older frames from the queue before adding new ones
        while not self.detection_queue.empty():
            try:
                self.detection_queue.get_nowait()  # Drop oldest frame
            except queue.Empty:
                break

        try:
            self.detection_queue.put_nowait((cam_id0, gray_cam0))
            self.detection_queue.put_nowait((cam_id1, gray_cam1))
        except queue.Full:
            logger.warning("Detection queue is full, dropping frame!")


    def display_combined(self, image_caption):
        """Combine both camera feeds into a single window and display."""
        if len(self.last_frames) >= 2:
            cam_ids = list(self.last_frames.keys())
            img_left = resize_for_display(self.last_frames[cam_ids[0]])
            img_right = resize_for_display(self.last_frames[cam_ids[1]])

            # **Concatenate both images side by side**
            combined_frame = np.hstack((img_left, img_right))

            # **Ensure OpenCV updates are smooth**
            if time.time() - self.last_update_time > 0.03:  # Limit to ~30 FPS
                cv2.imshow(image_caption, combined_frame)
                self.last_update_time = time.time()
        else:
            cv2.imshow(image_caption, create_dummy_frame())

    def display_results(self, image_caption):
        """Fetch detection results and display them."""
        while True:
            try:
                cam_id_res, gray_res, detections_res = self.detection_result_queue.get_nowait()
                if gray_res is None or detections_res is None:
                    self.latest_results.pop(cam_id_res, None)
                else:
                    self.latest_results[cam_id_res] = (gray_res, detections_res)
            except queue.Empty:
                break

        if self.latest_results:
            processed_frames = []
            for cid in sorted(self.latest_results.keys()):
                gray, detections = self.latest_results[cid]
                color_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                for det in detections:
                    tag_id = det['tag_id']
                    corners = det['corners']
                    for i, corner in enumerate(corners):
                        corner = corner.flatten().astype(int)
                        color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)][i]
                        cv2.circle(color_frame, tuple(corner), 8, color, -1)
                    for i in range(4):
                        start = tuple(corners[i].flatten().astype(int))
                        end = tuple(corners[(i + 1) % 4].flatten().astype(int))
                        cv2.line(color_frame, start, end, (255, 255, 255), 2)
                    center = np.mean(corners, axis=0).flatten().astype(int)
                    cv2.putText(color_frame, str(tag_id), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                processed_frames.append(resize_for_display(color_frame))

            if processed_frames:
                display_frame = np.concatenate(processed_frames, axis=1)
                cv2.imshow(image_caption, display_frame)
        else:
            cv2.imshow(image_caption, create_dummy_frame())
