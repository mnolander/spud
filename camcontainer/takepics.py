import copy
import queue
import threading
from typing import Optional

import cv2
import numpy

from vmbpy import *

FRAME_QUEUE_SIZE = 10
FRAME_HEIGHT = 3036
FRAME_WIDTH = 4024

def print_preamble():
    print('////////////////////////////////////////')
    print('/// VmbPy Multithreading Example ///////')
    print('////////////////////////////////////////\n')
    print(flush=True)

def add_camera_id(frame: Frame, cam_id: str) -> Frame:
    cv2.putText(frame.as_opencv_image(), 'Cam: {}'.format(cam_id), org=(0, 30), fontScale=1,
                color=255, thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)
    return frame

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

def set_nearest_value(cam: Camera, feat_name: str, feat_value: int):
    feat = cam.get_feature_by_name(feat_name)

    try:
        feat.set(feat_value)
    except VmbFeatureError:
        min_, max_ = feat.get_range()
        inc = feat.get_increment()

        if feat_value <= min_:
            val = min_
        elif feat_value >= max_:
            val = max_
        else:
            val = (((feat_value - min_) // inc) * inc) + min_

        feat.set(val)

        # msg = ('Camera {}: Failed to set value of Feature '{}' to '{}': '
        #        'Using nearest valid value '{}'. Note that, this causes resizing '
        #        'during processing, reducing the frame rate.')
        Log.get_instance().info(msg.format(cam.get_id(), feat_name, feat_value, val))

class FrameProducer(threading.Thread):
    def __init__(self, cam: Camera, frame_queue: queue.Queue):
        threading.Thread.__init__(self)

        self.log = Log.get_instance()
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

    def setup_camera(self):
        set_nearest_value(self.cam, 'Height', FRAME_HEIGHT)
        set_nearest_value(self.cam, 'Width', FRAME_WIDTH)

        try:
            self.cam.ExposureAuto.set('Off')
            self.cam.ExposureTime.set(16000)
            self.cam.Gain.set(0.000000001)
            self.cam.BinningHorizontal.set(2)
            self.cam.BinningVertical.set(2)
        except (AttributeError, VmbFeatureError):
            print("error")

        self.cam.set_pixel_format(PixelFormat.Mono8)

    def run(self):
        

        try:
            with self.cam:
                self.setup_camera()
                try:
                    self.cam.start_streaming(self)
                    self.killswitch.wait()
                finally:
                    self.cam.stop_streaming()
        except VmbCameraError:
            pass
        finally:
            try_put_frame(self.frame_queue, self.cam, None)

        
class FrameConsumer:
    def __init__(self, frame_queue: queue.Queue):
        self.log = Log.get_instance()
        self.frame_queue = frame_queue
        self.image_save_index = 0

    def run(self):
        IMAGE_CAPTION = 'Multithreading Example: Press <Enter> to exit, R to save image'
        KEY_CODE_ENTER = 13
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

                max_display_height = 1080
                max_display_width = 1920

                height, width = combined_image.shape[:2]
                scaling_factor = min(max_display_width / width, max_display_height / height)

                if scaling_factor < 1:
                    new_width = int(width * scaling_factor)
                    new_height = int(height * scaling_factor)
                    resized_combined_image = cv2.resize(combined_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                else:
                    resized_combined_image = combined_image

                cv2.imshow(IMAGE_CAPTION, resized_combined_image)
            else:
                cv2.imshow(IMAGE_CAPTION, create_dummy_frame())

            key = cv2.waitKey(10) & 0xFF
            if key == KEY_CODE_ENTER:
                cv2.destroyAllWindows()
                alive = False
            elif key == KEY_CODE_R:
                for cam_id, cv_image in cv_images.items():
                    filename = f'camera_{cam_id}_image_{self.image_save_index}.png'
                    cv2.imwrite(filename, cv_image)
                    print(f'Image saved as {filename}')
                self.image_save_index += 1

        

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
        log = Log.get_instance()
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
    app = Application()
    app.run()
