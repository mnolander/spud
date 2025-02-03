import queue
from frame_consumer import FrameConsumer
from camera import FrameProducer
from constants import FRAME_QUEUE_SIZE
from vmbpy import VmbSystem

class Application:
    def __init__(self):
        """Initialize the application with a shared frame queue."""
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.producers = {}

    def run(self):
        """Start the camera producers and the frame consumer."""
        consumer = FrameConsumer(self.frame_queue)  # Pass frame_queue here
        
        with VmbSystem.get_instance() as vmb:
            # Start producers for all currently detected cameras
            for cam in vmb.get_all_cameras():
                producer = FrameProducer(cam, self.frame_queue)
                self.producers[cam.get_id()] = producer

            # Start all producers
            for producer in self.producers.values():
                producer.start()

            # Run the consumer loop
            consumer.run()

            # Stop all producers when done
            for producer in self.producers.values():
                producer.stop()
                producer.join()
