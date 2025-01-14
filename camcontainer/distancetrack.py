import threading
import queue
import cv2
import numpy as np
import sys
import os
from vmbpy import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "enhanced_python_aprilgrid")))
from src.aprilgrid import Detector

# Constants
FRAME_QUEUE_SIZE = 20
TAG_SIZE = 0.02  # Tag size in meters (e.g., 2 cm)
FOCAL_LENGTH = 3940  # Example focal length in pixels
opencv_display_format = PixelFormat.Bgr8


def calculate_distance(tag_size, focal_length, tag_width_pixels):
    return (tag_size * focal_length) / tag_width_pixels


def setup_camera(cam: Camera):
    with cam:
        try:
            cam.ExposureAuto.set('Off')
            cam.ExposureTime.set(20000)
            cam.Gain.set(16)
            cam.BinningHorizontal.set(1)
            cam.BinningVertical.set(1)
        except (AttributeError, VmbFeatureError):
            pass


def setup_pixel_format(cam: Camera):
    cam_formats = cam.get_pixel_formats()
    cam_color_formats = intersect_pixel_formats(cam_formats, COLOR_PIXEL_FORMATS)
    convertible_color_formats = tuple(f for f in cam_color_formats
                                      if opencv_display_format in f.get_convertible_formats())

    cam_mono_formats = intersect_pixel_formats(cam_formats, MONO_PIXEL_FORMATS)
    convertible_mono_formats = tuple(f for f in cam_mono_formats
                                     if opencv_display_format in f.get_convertible_formats())

    if opencv_display_format in cam_formats:
        cam.set_pixel_format(opencv_display_format)
    elif convertible_color_formats:
        cam.set_pixel_format(convertible_color_formats[0])
    elif convertible_mono_formats:
        cam.set_pixel_format(convertible_mono_formats[0])
    else:
        raise RuntimeError("Camera does not support an OpenCV compatible format.")


class FrameHandler:
    def __init__(self, cam_id, frame_queue):
        self.cam_id = cam_id
        self.frame_queue = frame_queue

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        try:
            if frame.get_status() == FrameStatus.Complete:
                # Ensure the frame is in grayscale format
                gray_frame = frame.convert_pixel_format(opencv_display_format).as_numpy_ndarray()
                if gray_frame.ndim > 2:  # Convert to grayscale if not already
                    gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)
                self.frame_queue.put((self.cam_id, gray_frame))
            cam.queue_frame(frame)  # Requeue frame for further use
        except Exception as e:
            print(f"Error in FrameHandler: {e}")


class FrameProducer(threading.Thread):
    def __init__(self, cam: Camera, frame_queue: queue.Queue):
        super().__init__()
        self.cam = cam
        self.frame_queue = frame_queue
        self.handler = FrameHandler(cam.get_id(), frame_queue)
        self.running = True

    def run(self):
        with self.cam:
            setup_camera(self.cam)
            setup_pixel_format(self.cam)
            self.cam.start_streaming(handler=self.handler, buffer_count=10)

            while self.running:
                pass

            self.cam.stop_streaming()

    def stop(self):
        self.running = False


class FrameConsumer:
    def __init__(self, frame_queue):
        self.frame_queue = frame_queue
        self.detector = Detector("t16h5b1")
        self.running = True

    def run(self):
        cv2.namedWindow("Camera Streams", cv2.WINDOW_NORMAL)

        while self.running:
            frames = {}

            while not self.frame_queue.empty():
                cam_id, frame = self.frame_queue.get()
                frames[cam_id] = frame

            if frames:
                combined_image = None

                for cam_id, frame in frames.items():
                    processed_frame = self.detect_and_draw_apriltags(frame, cam_id)
                    if combined_image is None:
                        combined_image = processed_frame
                    else:
                        combined_image = np.hstack((combined_image, processed_frame))

                cv2.imshow("Camera Streams", combined_image)

            if cv2.waitKey(1) & 0xFF == 13:  # Enter key to exit
                self.running = False
                break

        cv2.destroyAllWindows()

    def detect_and_draw_apriltags(self, image, cam_id):
        try:
            detections = self.detector.detect(image)

            # Convert grayscale to color for display
            image_display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            for detection in detections:
                corners = detection.corners.astype(int)
                cv2.polylines(image_display, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
                center = tuple(detection.center.astype(int))
                cv2.circle(image_display, center, radius=5, color=(0, 0, 255), thickness=-1)

                tag_width_pixels = np.linalg.norm(corners[0] - corners[1])
                distance = calculate_distance(TAG_SIZE, FOCAL_LENGTH, tag_width_pixels)
                cv2.putText(image_display, f"ID: {detection.tag_id}, Dist: {distance:.2f}m",
                            (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            return image_display
        except Exception as e:
            print(f"Error during AprilTag detection for camera {cam_id}: {e}")
            # Return the input image as fallback
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


class MultiCameraApplication:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.producers = []

    def run(self):
        with VmbSystem.get_instance() as vmb:
            cameras = vmb.get_all_cameras()
            if not cameras:
                print("No cameras detected. Please connect at least one camera.")
                return

            for cam in cameras[:2]:  # Limit to two cameras
                producer = FrameProducer(cam, self.frame_queue)
                self.producers.append(producer)
                producer.start()

            consumer = FrameConsumer(self.frame_queue)
            consumer.run()

            for producer in self.producers:
                producer.stop()
                producer.join()


if __name__ == "__main__":
    app = MultiCameraApplication()
    app.run()