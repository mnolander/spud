import rospy
import tf
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
import queue
import threading
import logging

class ROSTopicPublisher(threading.Thread):
    """
    A ROS publisher thread that:
      - Listens to the `ros_result_queue` from `DetectorThread`
      - Converts 3D positions into ROS Transform messages
      - Publishes them to the `/tf` topic
    """

    def __init__(self, ros_result_queue: queue.Queue):
        super().__init__()
        self.ros_result_queue = ros_result_queue  # Queue from `DetectorThread`
        rospy.init_node('april_tag_tf_publisher', anonymous=True)
        self.tf_pub = rospy.Publisher('/tf', TFMessage, queue_size=10)
        self.killswitch = threading.Event()
        self.rate = rospy.Rate(30)  # 30 Hz

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ROSTopicPublisher")

    def stop(self):
        """Stop the publishing thread."""
        self.killswitch.set()

    def run(self):
        """Main loop that fetches detections from the queue and publishes to ROS."""
        while not rospy.is_shutdown() and not self.killswitch.is_set():
            try:
                self.logger.info("🔄 Waiting for tag results...")
                tag_results = self.ros_result_queue.get(timeout=0.5)

                if not tag_results:
                    self.logger.warning("⚠ No detections received. Queue is empty!")
                    continue

                self.logger.info(f"✅ Received {len(tag_results)} tags from queue.")

                tf_message = TFMessage()
                current_time = rospy.Time.now()

                for tag_id, (x, y, z), quaternion in tag_results:
                    self.logger.info(f"📌 Processing Tag {tag_id}: Position ({x:.3f}, {y:.3f}, {z:.3f})")

                    transform = TransformStamped()
                    transform.header.stamp = current_time
                    transform.header.frame_id = "stereo_camera"
                    transform.child_frame_id = f"tag_{tag_id}"

                    transform.transform.translation.x = x
                    transform.transform.translation.y = y
                    transform.transform.translation.z = z

                    transform.transform.rotation.x = quaternion[0]
                    transform.transform.rotation.y = quaternion[1]
                    transform.transform.rotation.z = quaternion[2]
                    transform.transform.rotation.w = quaternion[3]

                    tf_message.transforms.append(transform)

                if tf_message.transforms:
                    self.logger.info(f"✅ Publishing {len(tf_message.transforms)} tag(s) to /tf.")
                    self.tf_pub.publish(tf_message)

                self.rate.sleep()

            except queue.Empty:
                self.logger.warning("⚠ ros_result_queue is EMPTY. No detections received.")
                continue


if __name__ == '__main__':
    ros_result_queue = queue.Queue()
    ros_publisher = ROSTopicPublisher(ros_result_queue)

    # ✅ Simulate an AprilTag detection
    import time
    time.sleep(2)  # Wait before sending data

    # Simulated Data
    test_data = [
        (1, (0.5, 0.2, 1.0), (0.0, 0.0, 0.0, 1.0)),  # Tag 1
        (2, (1.2, -0.3, 0.8), (0.1, 0.2, 0.3, 0.9)),  # Tag 2
    ]
    
    ros_result_queue.put(test_data)  # Send to queue

    try:
        ros_publisher.run()
    except rospy.ROSInterruptException:
        pass
