import cv2
import numpy as np
import sys
import os

# If aprilgrid is under ../src:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "enhanced_python_aprilgrid", "src")))
from aprilgrid import Detector  

class StereoAprilTag:
    def __init__(self):
        # Stereo Camera Intrinsics
        self.intrinsics_cam0 = np.array([3315.158744682164, 3317.4423656635427, 1044.9919174051424, 769.2791390484149])
        self.intrinsics_cam1 = np.array([3303.1303751889086, 3304.4918413538717, 1021.6391642423999, 738.8164474717523])

        self.distortion_coeffs_cam0 = np.array([-0.26727533865087455, 0.6914547210589541, -0.00022631332732631944, 0.0006445742114670336])
        self.distortion_coeffs_cam1 = np.array([-0.25984247075537714, 0.609867968623631, 0.0008865986409872509, -0.000975406239237423])

        # Transformation matrix between cameras
        self.T_cn_cnm1 = np.array([
            [0.9958785322848419, -0.001883418317043302, 0.09067745954553587, -0.19167885361868625],
            [0.0021215517599516157, 0.9999945494395888, -0.0025298437192514665, -0.0007459795575273124],
            [-0.09067220054856458, 0.0027117939739249486, 0.9958770999581846, 0.006433227010261678],
            [0.0, 0.0, 0.0, 1.0]
        ])

        self.image_size = (2012, 1518)

        # Intrinsic camera matrices
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

       # this is gotten from the retify code
        self.P1 = np.array([
            [3.31096710e+03, 0.00000000e+00, 9.91145844e+02, 0.00000000e+00],
            [0.00000000e+00, 3.31096710e+03, 7.54056435e+02, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]
        ])

        self.P2 = np.array([
            [3.31096710e+03, 0.00000000e+00, 9.91145844e+02, -6.35004526e+02],
            [0.00000000e+00, 3.31096710e+03, 7.54056435e+02, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]
        ])

        # Initialize the AprilTag Detector
        self.apriltag_detector = Detector('t16h5b1')

    def detect_apriltags(self, image):
        """ Detect AprilTags in an image and return detected tag corners and IDs. """
        if image is None:
            print("Error: Image not loaded properly.")
            return None, []

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags
        detections = self.apriltag_detector.detect(gray)
        print(f"Detected {len(detections)} tags")

        tag_positions = {}

        for tag in detections:
            corners = np.array(tag.corners)  # Ensure corners are NumPy arrays
            center = np.mean(corners, axis=0).flatten()  # Compute center of tag

            # Store center position for 3D triangulation
            tag_positions[tag.tag_id] = center

            print(f"DEBUG: Detected tag ID {tag.tag_id} at position {tuple(map(int, center[:2]))}")

        return tag_positions  # Return detected tags with their positions

    def triangulate_3d_positions(self, left_pts, right_pts):
        """ Triangulate 3D positions from matched 2D points in left and right images. """
        # Convert to homogeneous coordinates
        left_pts_h = np.vstack((left_pts.T, np.ones((1, left_pts.shape[0]))))
        right_pts_h = np.vstack((right_pts.T, np.ones((1, right_pts.shape[0]))))

        # Triangulate 3D points
        points_4d_hom = cv2.triangulatePoints(self.P1, self.P2, left_pts_h[:2], right_pts_h[:2])

        # Convert from homogeneous to Euclidean coordinates
        points_3d = points_4d_hom[:3] / points_4d_hom[3]  # Normalize by dividing by w

        return points_3d.T  # Return as Nx3 array

    def compute_3d_positions(self, detections_left, detections_right):
        """ Match detected tags and compute 3D positions using stereo triangulation. """
        matched_ids = set(detections_left.keys()).intersection(detections_right.keys())

        if not matched_ids:
            print("No matching AprilTags found in both images.")
            return

        left_pts = np.array([detections_left[tag_id] for tag_id in matched_ids])
        right_pts = np.array([detections_right[tag_id] for tag_id in matched_ids])

        points_3d = self.triangulate_3d_positions(left_pts, right_pts)

        for tag_id, (x, y, z) in zip(matched_ids, points_3d):
            print(f"Tag ID {tag_id}: 3D Position (X: {x:.6f}, Y: {y:.6f}, Z: {z:.6f})")

    def process_images(self, left_img_path, right_img_path):
        """ Process stereo images for AprilTag detection and 3D triangulation. """
        img_left = cv2.imread(left_img_path)
        img_right = cv2.imread(right_img_path)

        if img_left is None or img_right is None:
            print("Error: Unable to load images.")
            return

        # Detect AprilTags
        detections_left = self.detect_apriltags(img_left)
        detections_right = self.detect_apriltags(img_right)

        # Compute 3D positions
        self.compute_3d_positions(detections_left, detections_right)

        # Display images
        cv2.imshow("Left Camera - AprilTag Detection", img_left)
        cv2.imshow("Right Camera - AprilTag Detection", img_right)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        



# Example usage
if __name__ == "__main__":
    stereo_apriltag = StereoAprilTag()
    stereo_apriltag.process_images('rectifiedT5_left3.jpg', 'rectifiedT5_right3.jpg')
