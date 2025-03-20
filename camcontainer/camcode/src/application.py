import queue
from frame_consumer import FrameConsumer
from camera import FrameProducer
from constants import FRAME_QUEUE_SIZE
from vmbpy import VmbSystem

class Application:
    def __init__(self):
        """Initialize the application with a shared frame queue."""
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.producers = {} # Dictionary to store the camera producers

    def run(self):
        """Start the camera producers and the frame consumer."""
        consumer = FrameConsumer(self.frame_queue) 
        
        with VmbSystem.get_instance() as vmb:
            for cam in vmb.get_all_cameras():
                producer = FrameProducer(cam, self.frame_queue)
                self.producers[cam.get_id()] = producer

            for producer in self.producers.values():
                producer.start()

            consumer.run()

            for producer in self.producers.values():
                producer.stop()
                producer.join()
