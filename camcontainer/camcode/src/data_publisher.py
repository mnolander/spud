#!/usr/bin/env python

import rospy
import tf
import subprocess
import re
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped


def extract_position(output_data):
    """
    Extracts X, Y, Z values from the first line.
    Example:
    Tag ID 0: 3D Position (X: -0.038496, Y: -0.084894, Z: 1.044426)
    """
    match = re.search(r'X:\s*([-0-9.]+),\s*Y:\s*([-0-9.]+),\s*Z:\s*([-0-9.]+)', output_data)
    if match:
        return float(match.group(1)), float(match.group(2)), float(match.group(3))
    return None

def extract_quaternion(output_data):
    """
    Extracts quaternion (qx, qy, qz, qw) from the second line.
    Example:
    [-0.61552422  0.78468067  0.06577317  0.03286444]
    """
    match = re.search(r'\[([-0-9.e\s]+)\]', output_data)
    if match:
        quaternion_values = list(map(float, match.group(1).split()))
        if len(quaternion_values) == 4:
            return tuple(quaternion_values)
    return None

def talker():
    rospy.init_node('tf_publisher', anonymous=True)
    
    tf_pub = rospy.Publisher('/tf', TFMessage, queue_size=100)

    rospy.loginfo("Starting tf_publisher.py...")

    process = subprocess.Popen(
        ["python3", "/home/ubuntu/capstone/dronecamtoolbox/camcontainer/camcode/src/main.py"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=10
    )

    rate = rospy.Rate(200)  # 35 Hz

    while not rospy.is_shutdown():
        try:
            # Read first line (position data)
            position_line = process.stdout.readline().strip()

            if not position_line:
                continue  # Skip if empty

            rospy.loginfo(f"🔹 Received Position: {position_line}")
            position = extract_position(position_line)

            if position is None:
                rospy.logwarn(f"⚠ Invalid Position Data: {position_line}")
                continue  # Skip to next iteration

            # Read second line (quaternion data)
            quaternion_line = process.stdout.readline().strip()

            if not quaternion_line:
                continue  # Skip if empty

            rospy.loginfo(f"🔹 Received Quaternion: {quaternion_line}")
            quaternion = extract_quaternion(quaternion_line)

            if quaternion is None:
                rospy.logwarn(f"⚠ Invalid Quaternion Data: {quaternion_line}")
                continue  # Skip to next iteration


            # If both position and quaternion are valid, publish TF
            x, y, z = map(lambda v: round(v, 4), position)
            qx, qy, qz, qw = map(lambda v: round(v, 3), quaternion)


            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = "base_link"
            transform.child_frame_id = "camera_link"

            transform.transform.translation.x = x
            transform.transform.translation.y = y
            transform.transform.translation.z = z

            transform.transform.rotation.x = qx
            transform.transform.rotation.y = qy
            transform.transform.rotation.z = qz
            transform.transform.rotation.w = qw

            transform_odom = TransformStamped()
            transform_odom.header.stamp = rospy.Time.now()


            transform_odom.header.frame_id = "camera_link"
            transform_odom.child_frame_id = "camera_odom_frame"

            transform_odom.transform.translation.x = x
            transform_odom.transform.translation.y = y
            transform_odom.transform.translation.z = z

            transform_odom.transform.rotation.x = qx
            transform_odom.transform.rotation.y = qy
            transform_odom.transform.rotation.z = qz
            transform_odom.transform.rotation.w = qw

            tf_message = TFMessage(transforms=[transform, transform_odom])
            rospy.loginfo(f"Publishing TF: {tf_message}")
            tf_pub.publish(tf_message)

        except Exception as e:
            rospy.logerr(f" Exception: {e}")

    rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass