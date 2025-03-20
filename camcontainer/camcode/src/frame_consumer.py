import queue
import cv2
import logging
from cam_detector import DetectorThread
from constants import FRAME_QUEUE_SIZE, NUM_DETECTOR_THREADS, DETECTION_FRAME_SKIP
from vmbpy import *
from frame_processing import resize_for_display
from utils import create_dummy_frame
from stereo_processor import StereoProcessor
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cv2.setUseOptimized(True)
cv2.setNumThreads(NUM_DETECTOR_THREADS)  

class FrameConsumer:
    """
    Main consumer class that:
      1) Pulls frames from the frame_queue.
      2) Hands them off to the parallel detector threads (detection_queue),
         optionally skipping frames to reduce CPU load.
      3) Receives detection results (detection_result_queue).
      4) Draws detection overlays and displays in a window.
    """

    def __init__(self, frame_queue: queue.Queue, num_detector_threads: int = 2):
        self.frame_queue = frame_queue

        # Detections queues for parallel processing of frames
        self.detection_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE*2)
        self.detection_result_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE*2)

        self.detector_threads = []
        for i in range(num_detector_threads):
            dt = DetectorThread(self.detection_queue, self.detection_result_queue)
            dt.start()
            self.detector_threads.append(dt)

        self.stereo_processor = StereoProcessor()
        self.camera_data = {}
        self.last_frames = {}
        self.latest_results = {}
        self.frame_counts = {}

    def log_frame_info(self, cam_id: str, frame: 'Frame'):
        """Log frame resolution."""
        resolution = (frame.get_width(), frame.get_height())

    def run(self):
        """Main loop: fetch frames, pass to detector, fetch results, draw and display."""
        IMAGE_CAPTION = 'DroneCam Multicam View: Press <Enter> to exit'
        KEY_CODE_ENTER = 13

        alive = True
        cv2.namedWindow(IMAGE_CAPTION, cv2.WINDOW_AUTOSIZE)

        while alive:
            # Retrieve new frames from the queue
            try:
                cam_id, frame = self.frame_queue.get(timeout=0.01)
                if frame:
                    self.log_frame_info(cam_id, frame)
                    self.last_frames[cam_id] = frame

                    if cam_id not in self.frame_counts:
                        self.frame_counts[cam_id] = 0
                    self.frame_counts[cam_id] += 1

                    # Only offload every Nth frame to the detector to reduce CPU load
                    if self.frame_counts[cam_id] % DETECTION_FRAME_SKIP == 0:
                        try:
                            self.detection_queue.put_nowait((cam_id, frame))
                        except queue.Full:
                            pass
                else:
                    self.last_frames.pop(cam_id, None)
                    self.latest_results.pop(cam_id, None)
                    try:
                        self.detection_queue.put_nowait((cam_id, None))
                    except queue.Full:
                        pass

            except queue.Empty:
                pass

            while True:
                # Retrieve detection results from the queue
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

                    # Draw detection overlays for AprilTags Corner points
                    for det in detections:
                        tag_id = det['tag_id']
                        corners = det['corners'] 

                        for i, corner in enumerate(corners):
                            corner = corner.flatten().astype(int)
                            if i == 0:
                                color = (0, 0, 255)      
                            elif i == 1:
                                color = (0, 255, 0)      
                            elif i == 2:
                                color = (255, 0, 0)      
                            else:
                                color = (0, 255, 255)    
                            cv2.circle(color_frame, tuple(corner), 10, color, -1)

                        # Draw lines between the corners
                        for i in range(4):
                            start = tuple(corners[i].flatten().astype(int))
                            end = tuple(corners[(i + 1) % 4].flatten().astype(int))
                            cv2.line(color_frame, start, end, (255, 255, 255), 2)

                        center = np.mean(corners, axis=0).flatten().astype(int)
                        cv2.putText(color_frame, str(tag_id), tuple(center),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    display_frame = resize_for_display(color_frame)
                    processed_frames.append(display_frame)

                # Display the frames in a single window
                if processed_frames:
                    display_frame = np.concatenate(processed_frames, axis=1)
                    cv2.imshow(IMAGE_CAPTION, display_frame)
            else:
                cv2.imshow(IMAGE_CAPTION, create_dummy_frame())

            if KEY_CODE_ENTER == cv2.waitKey(1):
                cv2.destroyAllWindows()
                alive = False
        for dt in self.detector_threads:
            dt.stop()
        for dt in self.detector_threads:
            dt.join()