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
        self.intrinsics_cam0 = np.array([3431.6333053510584, 3425.4233661235844, 1037.0631103881203, 814.2748160563247])
        self.intrinsics_cam1 = np.array([3450.443379487295, 3442.6715206818894, 1000.8126349532237, 768.292008152334])

        self.distortion_coeffs_cam0 = np.array([-0.29763392127051924, 0.8835065755916732, -0.0023637256530460084, -0.0007121489154046612])
        self.distortion_coeffs_cam1 = np.array([-0.2760132481960459, 0.7026370095608109, -0.0015314823046884355, -0.001072650889485783])

        # Transformation matrix between cameras
        self.T_cn_cnm1 = np.array([
            [0.9954706102064841, -0.002938725639798721, 0.09502435533453286, -0.1898177118318758],
            [0.002702420629833145, 0.999992928337077, 0.0026153773296546125, -0.001517588623992596],
            [-0.09503136923073889, -0.002346735488078846, 0.9954715122466737, 0.01757994327060201],
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
            [3.43404744e+03, 0.00000000e+00, 1.18656602e+03, 0.00000000e+00],
            [0.00000000e+00, 3.43404744e+03, 7.89917580e+02, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]
        ])

        self.P2 = np.array([
            [3.43404744e+03, 0.00000000e+00, 1.18656602e+03, -6.54653399e+02],
            [0.00000000e+00, 3.43404744e+03, 7.89917580e+02, 0.00000000e+00],
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
    stereo_apriltag.process_images('rectified_left3.jpg', 'rectified_right3.jpg')
