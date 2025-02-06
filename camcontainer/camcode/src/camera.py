import queue
import threading
from vmbpy import *

from constants import FRAME_HEIGHT, FRAME_WIDTH

def try_put_frame(q: queue.Queue, cam: 'Camera', frame: 'Frame'):
    """Attempt to put a frame into the queue without blocking."""
    try:
        q.put_nowait((cam.get_id(), frame))
    except queue.Full:
        pass

def set_nearest_value(cam: 'Camera', feat_name: str, feat_value: int):
    """Set a camera feature to the nearest valid increment if needed."""
    feat = cam.get_feature_by_name(feat_name)
    try:
        feat.set(feat_value)
    except VmbFeatureError:
        min_, max_ = feat.get_range()
        inc = feat.get_increment()
        val = max(min_, min(max_, (((feat_value - min_) // inc) * inc) + min_))
        feat.set(val)

class FrameProducer(threading.Thread):
    """
    Captures frames from a camera in a separate thread.
    For each completed frame, adds it to the shared frame_queue.
    """

    def __init__(self, cam: 'Camera', frame_queue: queue.Queue):
        super().__init__()
        self.cam = cam
        self.frame_queue = frame_queue
        self.killswitch = threading.Event()

    def __call__(self, cam: 'Camera', stream: 'Stream', frame: 'Frame'):
        """Callback for camera frames."""
        if frame.get_status() == FrameStatus.Complete:
            try_put_frame(self.frame_queue, cam, frame)
        cam.queue_frame(frame)

    def stop(self):
        """Stop this thread's loop."""
        self.killswitch.set()

    def setup_camera(self):
        """Configure camera parameters."""
        set_nearest_value(self.cam, 'Height', FRAME_HEIGHT)
        set_nearest_value(self.cam, 'Width', FRAME_WIDTH)

        try:
            self.cam.ExposureAuto.set('Off')
            self.cam.ExposureTime.set(20000)
            self.cam.Gain.set(0.000000001)
            self.cam.BinningHorizontal.set(2)
            self.cam.BinningVertical.set(2)

            self.cam.AcquisitionFrameRateEnable.set(True)
            self.cam.AcquisitionFrameRate.set(35.0)

        except (AttributeError, VmbFeatureError):
            pass

        # Force Mono8 pixel format
        self.cam.set_pixel_format(PixelFormat.Mono8)

    def run(self):
        """Main thread loop: open camera, start streaming until killswitch is set."""
        with self.cam:
            self.setup_camera()
            try:
                self.cam.start_streaming(self)
                self.killswitch.wait()
            finally:
                self.cam.stop_streaming()
        # Signal to consumer that camera is missing by sending None
        try_put_frame(self.frame_queue, self.cam, None)
