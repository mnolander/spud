#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped

def pose_publisher():
    # Create a publisher on the '/pose' topic.
    pub = rospy.Publisher('/pose', PoseStamped, queue_size=10)
    
    # Initialize the node. The 'anonymous=True' option means that ROS will generate a unique name for the node.
    rospy.init_node('pose_publisher', anonymous=True)
    
    # Set the loop rate (e.g., 10 Hz)
    rate = rospy.Rate(10)
    
    while not rospy.is_shutdown():
        # Create and fill in a PoseStamped message.
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()  # Use current time for timestamp
        pose_msg.header.frame_id = "camera_odom_frame"
        
        # Replace the following with your dynamically computed values if needed.
        pose_msg.pose.position.x = -0.7516490746879579
        pose_msg.pose.position.y =  1.8715157897292756
        pose_msg.pose.position.z =  1.5548264342916485
        
        pose_msg.pose.orientation.x = -0.1564322211607215
        pose_msg.pose.orientation.y = -0.3760698501604714
        pose_msg.pose.orientation.z =  0.4419974505059653
        pose_msg.pose.orientation.w =  0.7992112872884307

        # Log the published message (optional)
        rospy.loginfo("Publishing pose: %s", pose_msg)
        
        # Publish the message to the topic.
        pub.publish(pose_msg)
        
        # Sleep to maintain the loop rate.
        rate.sleep()

if __name__ == '__main__':
    try:
        pose_publisher()
    except rospy.ROSInterruptException:
        pass
