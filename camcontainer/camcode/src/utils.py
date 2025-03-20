import numpy as np
import cv2
import math
from scipy.spatial.transform import Rotation as R

def print_preamble():
    """Prints the preamble for the application."""
    print('////////////////////////////////////////')
    print('/// DroneCam Multicam View ///////')
    print('////////////////////////////////////////\n')
    print(flush=True)

def create_dummy_frame() -> np.ndarray:
    """Creates a dummy frame when no camera is available."""
    cv_frame = np.zeros((50, 640, 1), np.uint8)
    cv2.putText(
        cv_frame,
        'No Stream available. Please connect a Camera.',
        org=(30, 30),
        fontScale=1,
        color=255,
        thickness=1,
        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL
    )
    return cv_frame

def rotation_matrix_to_quaternion(rotation_matrix):
    """Converts a 3x3 rotation matrix to a quaternion."""
    return R.from_matrix(rotation_matrix).as_quat().tolist()

def compute_distance(intrinsics, tag_size, corners):
    """
    Compute the distance of the AprilTag from the camera.

    Args:
        intrinsics (list): Intrinsic parameters of the camera [fx, fy, cx, cy].
        tag_size (float): Physical size of the AprilTag in meters.
        corners (list): Detected corner coordinates of the tag.

    Returns:
        float: Distance from the camera to the tag in meters, or None if invalid input.
    """
    if len(corners) < 4:
        return None

    fx, fy, cx, cy = intrinsics

    def euclidean_distance(pt1, pt2):
        """Computes Euclidean distance between two points."""
        return math.sqrt((pt1[0][0] - pt2[0][0]) ** 2 + (pt1[0][1] - pt2[0][1]) ** 2)

    try:
        d1 = euclidean_distance(corners[0], corners[2])
        d2 = euclidean_distance(corners[1], corners[3])
        avg_pixel_distance = (d1 + d2) / 2

        distance = (fx * tag_size) / avg_pixel_distance
        return distance
    except IndexError:
        return None

def compute_pose(intrinsics, tag_size, corners, distortion_coeffs):
    """
    Compute the pose of the AprilTag from the camera.

    Args:
        intrinsics (list): Intrinsic parameters of the camera [fx, fy, cx, cy].
        tag_size (float): Physical size of the AprilTag in meters.
        corners (list): Detected corner coordinates of the tag.
        distortion_coeffs (list): Distortion coefficients of the camera.

    Returns:
        dict: Pose data including position, orientation, and distance.
    """
    if len(corners) < 4:
        print("Insufficient corners detected.")
        return None

    fx, fy, cx, cy = intrinsics
    half_size = tag_size / 2

    # Real-world coordinates of the AprilTag corners
    obj_points = np.array([
        [-half_size, -half_size, 0], 
        [ half_size, -half_size, 0],
        [ half_size,  half_size, 0], 
        [-half_size,  half_size, 0]
    ], dtype=np.float32)

    corners_2d = np.array([corner.flatten() for corner in corners], dtype=np.float32)

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.array(distortion_coeffs, dtype=np.float32)
    ret, rvec, tvec = cv2.solvePnP(obj_points, corners_2d, camera_matrix, dist_coeffs)

    if not ret:
        print("Pose computation failed: solvePnP could not find a solution")
        return None

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    distance = np.linalg.norm(tvec)

    pose_data = {
        "position": {
            "x": float(tvec[0][0]),
            "y": float(tvec[1][0]),
            "z": float((tvec[2][0])/2)
        },
        "orientation": {
            "x": quaternion[0],
            "y": quaternion[1],
            "z": quaternion[2],
            "w": quaternion[3]
        },
        "distance": float(distance)
    }

    return pose_data
