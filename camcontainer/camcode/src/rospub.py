import rospy
import tf
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
import queue

class ROSTopicPublisher:
    def __init__(self, detection_result_queue: queue.Queue):
        self.detection_result_queue = detection_result_queue
        rospy.init_node('april_tag_tf_publisher')

        # Create a publisher for the /tf topic
        self.tf_pub = rospy.Publisher('/tf', TFMessage, queue_size=10)

    def publish_detections(self):
        rate = rospy.Rate(30)  # 30 Hz

        while not rospy.is_shutdown():
            try:
                # Get detections from DetectorThread
                cam_id, _, detections, tag_results = self.detection_result_queue.get(timeout=0.1)

                if tag_results is None or len(tag_results) == 0:
                    continue  # No detections

                tf_message = TFMessage()
                current_time = rospy.Time.now()

                for tag_id, (x, y, z), quaternion in tag_results:
                    transform = TransformStamped()
                    transform.header.stamp = current_time
                    transform.header.frame_id = "stereo_camera"  # Parent frame
                    transform.child_frame_id = f"tag_{tag_id}"  # Unique frame for each tag

                    # ✅ Publish actual computed 3D position
                    transform.transform.translation.x = x
                    transform.transform.translation.y = y
                    transform.transform.translation.z = z

                    # ✅ Publish actual computed quaternion rotation
                    transform.transform.rotation.x = quaternion[0]
                    transform.transform.rotation.y = quaternion[1]
                    transform.transform.rotation.z = quaternion[2]
                    transform.transform.rotation.w = quaternion[3]

                    # Add transform to message
                    tf_message.transforms.append(transform)

                # ✅ Publish only if there are valid detections
                if tf_message.transforms:
                    self.tf_pub.publish(tf_message)

                rate.sleep()

            except queue.Empty:
                continue

if __name__ == '__main__':
    detection_result_queue = queue.Queue()
    ros_publisher = ROSTopicPublisher(detection_result_queue)

    try:
        ros_publisher.publish_detections()
    except rospy.ROSInterruptException:
        pass
