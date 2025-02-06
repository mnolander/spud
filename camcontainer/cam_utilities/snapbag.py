#This script is for showing live camera feed for multiple cameras using Multithreading
import copy
import queue
import threading
from typing import Optional

import cv2
import numpy

from vmbpy import *

# Import ROS modules
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rosbag
import time  # For timestamping the bag files

FRAME_QUEUE_SIZE = 10
FRAME_HEIGHT = 3036
FRAME_WIDTH = 4024
BURST_COUNT = 3  # Number of photos in a burst

def print_preamble():
    print('////////////////////////////////////////')
    print('/// VmbPy Multithreading Example ///////')
    print('////////////////////////////////////////\n')
    print(flush=True)

def resize_if_required(frame: Frame) -> numpy.ndarray:
    cv_frame = frame.as_opencv_image()

    if (frame.get_height() != FRAME_HEIGHT) or (frame.get_width() != FRAME_WIDTH):
        cv_frame = cv2.resize(cv_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        cv_frame = cv_frame[..., numpy.newaxis]

    return cv_frame

def create_dummy_frame() -> numpy.ndarray:
    cv_frame = numpy.zeros((50, 640, 1), numpy.uint8)
    cv_frame[:] = 0

    cv2.putText(cv_frame, 'No Stream available. Please connect a Camera.', org=(30, 30),
                fontScale=1, color=255, thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)

    return cv_frame

def try_put_frame(q: queue.Queue, cam: Camera, frame: Optional[Frame]):
    try:
        q.put_nowait((cam.get_id(), frame))
    except queue.Full:
        pass

class FrameProducer(threading.Thread):
    def __init__(self, cam: Camera, frame_queue: queue.Queue):
        threading.Thread.__init__(self)

        self.cam = cam
        self.frame_queue = frame_queue
        self.killswitch = threading.Event()

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            if not self.frame_queue.full():
                frame_cpy = copy.deepcopy(frame)
                try_put_frame(self.frame_queue, cam, frame_cpy)

        cam.queue_frame(frame)

    def stop(self):
        self.killswitch.set()

    def run(self):
        with self.cam:
            self.cam.set_pixel_format(PixelFormat.Mono8)
            self.cam.start_streaming(self)
            self.killswitch.wait()
            self.cam.stop_streaming()

        try_put_frame(self.frame_queue, self.cam, None)

class FrameConsumer:
    def __init__(self, frame_queue: queue.Queue):
        self.frame_queue = frame_queue
        self.bridge = CvBridge()
        self.burst_mode = False
        self.bag = None
        self.frames_to_save = []

    def start_bag(self):
        if not self.bag:
            bag_filename = time.strftime("%Y%m%d-%H%M%S") + '_burst_session.bag'
            self.bag = rosbag.Bag(bag_filename, 'w')
            print(f'Started recording to {bag_filename}')

    def save_burst_to_bag(self):
        if self.frames_to_save:
            for cam_id, frame in self.frames_to_save:
                ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="mono8")
                ros_image.header.stamp = rospy.Time.now()
                topic_name = f'/camera/{cam_id}/image'
                self.bag.write(topic_name, ros_image, ros_image.header.stamp)
            print(f'Saved burst of {len(self.frames_to_save)} frames to bag.')
            self.frames_to_save.clear()

    def stop_bag(self):
        if self.bag:
            self.bag.close()
            print('Stopped recording and closed the bag file.')
            self.bag = None

    def run(self):
        IMAGE_CAPTION = 'Multithreading Example: Press <E> to exit, R for burst'
        KEY_CODE_E = ord('e')
        KEY_CODE_R = ord('r')

        frames = {}
        alive = True

        cv2.namedWindow(IMAGE_CAPTION, cv2.WINDOW_NORMAL)

        while alive:
            frames_left = self.frame_queue.qsize()
            while frames_left:
                try:
                    cam_id, frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break

                if frame:
                    frames[cam_id] = frame
                else:
                    frames.pop(cam_id, None)

                frames_left -= 1

            if len(frames) >= 1:
                cv_images = {cam_id: resize_if_required(frames[cam_id]) for cam_id in sorted(frames.keys())}
                combined_image = numpy.concatenate(list(cv_images.values()), axis=1)
                cv2.imshow(IMAGE_CAPTION, combined_image)

                if self.burst_mode:
                    for cam_id, cv_image in cv_images.items():
                        self.frames_to_save.append((cam_id, cv_image))
                    if len(self.frames_to_save) >= BURST_COUNT:
                        self.save_burst_to_bag()
                        self.burst_mode = False
            else:
                cv2.imshow(IMAGE_CAPTION, create_dummy_frame())

            key = cv2.waitKey(10) & 0xFF
            if key == KEY_CODE_E:
                alive = False
                self.stop_bag()
                cv2.destroyAllWindows()
            elif key == KEY_CODE_R:
                print("Burst mode activated")
                self.burst_mode = True
                self.start_bag()

        if self.bag:
            self.stop_bag()

class Application:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.producers = {}
        self.producers_lock = threading.Lock()

    def __call__(self, cam: Camera, event: CameraEvent):
        if event == CameraEvent.Detected:
            with self.producers_lock:
                self.producers[cam.get_id()] = FrameProducer(cam, self.frame_queue)
                self.producers[cam.get_id()].start()

        elif event == CameraEvent.Missing:
            with self.producers_lock:
                if cam.get_id() in self.producers:
                    producer = self.producers.pop(cam.get_id())
                    producer.stop()
                    producer.join()

    def run(self):
        consumer = FrameConsumer(self.frame_queue)

        vmb = VmbSystem.get_instance()
        vmb.enable_log(LOG_CONFIG_INFO_CONSOLE_ONLY)

        with vmb:
            cameras = vmb.get_all_cameras()
            if len(cameras) < 2:
                print("Please connect at least two cameras.")
                return

            for cam in cameras:
                self.producers[cam.get_id()] = FrameProducer(cam, self.frame_queue)

            with self.producers_lock:
                for producer in self.producers.values():
                    producer.start()

            vmb.register_camera_change_handler(self)
            consumer.run()
            vmb.unregister_camera_change_handler(self)

            with self.producers_lock:
                for producer in self.producers.values():
                    producer.stop()
                for producer in self.producers.values():
                    producer.join()

if __name__ == '__main__':
    print_preamble()
    rospy.init_node('multi_camera_burst_recorder', anonymous=True)
    app = Application()
    app.run()
