import cv2
import numpy as np
import sys
import os

# If aprilgrid is under ../src:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "enhanced_python_aprilgrid", "src")))
from aprilgrid import Detector  

class StereoAprilTag:
    def __init__(self):
        # Stereo Calibration Parameters (same as before)
        self.intrinsics_cam0 = np.array([3431.6333053510584, 3425.4233661235844, 1037.0631103881203, 814.2748160563247])
        self.intrinsics_cam1 = np.array([3450.443379487295, 3442.6715206818894, 1000.8126349532237, 768.292008152334])

        self.distortion_coeffs_cam0 = np.array([-0.29763392127051924, 0.8835065755916732, -0.0023637256530460084, -0.0007121489154046612])
        self.distortion_coeffs_cam1 = np.array([-0.2760132481960459, 0.7026370095608109, -0.0015314823046884355, -0.001072650889485783])

        self.image_size = (2012, 1518)

        self.K1 = np.array([
            [self.intrinsics_cam0[0], 0, self.intrinsics_cam0[2]],
            [0, self.intrinsics_cam0[1], self.intrinsics_cam0[3]],
            [0, 0, 1]
            ])

        self.K2 = np.array([
            [self.intrinsics_cam1[0], 0, self.intrinsics_cam1[2]],
            [0, self.intrinsics_cam1[1], self.intrinsics_cam1[3]],
            [0, 0, 1]
            ])


        # AprilTag Detector
        self.apriltag_detector = Detector('t16h5b1')

    def detect_apriltags(self, image):
        """ Detect AprilTags in an image and return detected tag corners and IDs. """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.apriltag_detector.detect(gray)

        for tag in detections:
            corners = np.int32(tag.corners)  # Convert corners to int format

            # Draw bounding box
            for i in range(4):
                pt1 = tuple(map(int, corners[i]))
                pt2 = tuple(map(int, corners[(i+1) % 4]))
                cv2.line(image, pt1, pt2, (0, 255, 0), 2)

            # Draw tag ID
            cv2.putText(image, f"ID: {tag.tag_id}", tuple(map(int, corners[0])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        return image, detections

    def process_images(self, left_img_path, right_img_path):
        # Load images
        img_left = cv2.imread(left_img_path)
        img_right = cv2.imread(right_img_path)

        if img_left is None or img_right is None:
            print("Error loading images")
            return

        # Resize images if needed
        img_left = cv2.resize(img_left, self.image_size)
        img_right = cv2.resize(img_right, self.image_size)

        # Detect AprilTags
        img_left_detected, detections_left = self.detect_apriltags(img_left)
        img_right_detected, detections_right = self.detect_apriltags(img_right)

        # Show results
        cv2.imshow("Left Camera - AprilTag Detection", img_left_detected)
        cv2.imshow("Right Camera - AprilTag Detection", img_right_detected)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return detections_left, detections_right

# Example usage
if __name__ == "__main__":
    stereo_apriltag = StereoAprilTag()
    detections_left, detections_right = stereo_apriltag.process_images('rectified_left.jpg', 'rectified_right.jpg')
