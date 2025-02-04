import cv2
import numpy as np
import threading
import sys
import os

# If aprilgrid is under ../src:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "enhanced_python_aprilgrid", "src")))
from aprilgrid import Detector  

class CameraCalibration:
    def __init__(self, primary_intrinsics, secondary_intrinsics, primary_distortion, secondary_distortion, extrinsics):
        self.primary_intrinsics = primary_intrinsics
        self.secondary_intrinsics = secondary_intrinsics
        self.primary_distortion = primary_distortion
        self.secondary_distortion = secondary_distortion
        self.extrinsics = extrinsics
        self.R = extrinsics[:3, :3]
        self.T = extrinsics[:3, 3]

class StereoProcessor:
    def __init__(self):
        self.lock = threading.Lock()  # Thread synchronization lock
        self.rectified_frames = {}  # Stores the latest rectified frames

        # Camera intrinsic parameters
        intrinsics_cam0 = np.array([3315.158744682164, 3317.4423656635427, 1044.9919174051424, 769.2791390484149])
        intrinsics_cam1 = np.array([3303.1303751889086, 3304.4918413538717, 1021.6391642423999, 738.8164474717523])

        # Transformation matrix between cameras
        extrinsics = np.array([
            [0.9958785322848419, -0.001883418317043302, 0.09067745954553587, -0.19167885361868625],
            [0.0021215517599516157, 0.9999945494395888, -0.0025298437192514665, -0.0007459795575273124],
            [-0.09067220054856458, 0.0027117939739249486, 0.9958770999581846, 0.006433227010261678],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # Distortion coefficients
        distortion_coeffs_cam0 = np.array([-0.26727533865087455, 0.6914547210589541, -0.00022631332732631944, 0.0006445742114670336])
        distortion_coeffs_cam1 = np.array([-0.25984247075537714, 0.609867968623631, 0.0008865986409872509, -0.000975406239237423])

        self.calibration = CameraCalibration(intrinsics_cam0, intrinsics_cam1, distortion_coeffs_cam0, distortion_coeffs_cam1, extrinsics)

        self.image_size = (1518, 2008)

        # Intrinsic camera matrices
        self.K1 = np.array([
            [self.calibration.primary_intrinsics[0], 0, self.calibration.primary_intrinsics[2]],
            [0, self.calibration.primary_intrinsics[1], self.calibration.primary_intrinsics[3]],
            [0, 0, 1]
        ])
        
        self.K2 = np.array([
            [self.calibration.secondary_intrinsics[0], 0, self.calibration.secondary_intrinsics[2]],
            [0, self.calibration.secondary_intrinsics[1], self.calibration.secondary_intrinsics[3]],
            [0, 0, 1]
        ])

        # Stereo rectification
        self.R1_rect, self.R2_rect, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.K1, self.calibration.primary_distortion, self.K2, self.calibration.secondary_distortion, self.image_size, self.calibration.R, self.calibration.T, flags=cv2.CALIB_ZERO_DISPARITY
        )

        # Rectification maps
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(self.K1, self.calibration.primary_distortion, self.R1_rect, self.P1, self.image_size, cv2.CV_32F)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(self.K2, self.calibration.secondary_distortion, self.R2_rect, self.P2, self.image_size, cv2.CV_32F)

    def rectify_frames_async(self, frame_left, frame_right, cam_id1, cam_id2):
        """Runs rectification in a separate thread."""
        threading.Thread(target=self.rectify_frames, args=(frame_left, frame_right, cam_id1, cam_id2), daemon=True).start()

    def rectify_frames(self, frame_left, frame_right, cam_id1, cam_id2):
        """Rectifies stereo frames and stores them in a shared dictionary."""
        
        # ✅ Ensure frames are NumPy arrays
        if not isinstance(frame_left, np.ndarray):
            print(f"⚠️ Warning: Frame from {cam_id1} is not a valid NumPy array!")
            return
        if not isinstance(frame_right, np.ndarray):
            print(f"⚠️ Warning: Frame from {cam_id2} is not a valid NumPy array!")
            return

        # ✅ Ensure frames are not empty
        if frame_left.size == 0 or frame_right.size == 0:
            print(f"⚠️ Warning: Frame from {cam_id1} or {cam_id2} is empty!")
            return

        # ✅ Convert to grayscale if needed (remap requires single channel)
        if len(frame_left.shape) == 3 and frame_left.shape[2] == 3:
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        if len(frame_right.shape) == 3 and frame_right.shape[2] == 3:
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        # ✅ Resize frames to match calibration size if needed
        if frame_left.shape[:2] != self.image_size:
            print(f"⚠️ Resizing {cam_id1} from {frame_left.shape[:2]} to {self.image_size}")
            frame_left = cv2.resize(frame_left, self.image_size, interpolation=cv2.INTER_LINEAR)
        if frame_right.shape[:2] != self.image_size:
            print(f"⚠️ Resizing {cam_id2} from {frame_right.shape[:2]} to {self.image_size}")
            frame_right = cv2.resize(frame_right, self.image_size, interpolation=cv2.INTER_LINEAR)

        # ✅ Apply rectification
        rectified_left = cv2.remap(frame_left, self.map1x, self.map1y, interpolation=cv2.INTER_LINEAR)
        rectified_right = cv2.remap(frame_right, self.map2x, self.map2y, interpolation=cv2.INTER_LINEAR)

        # Store rectified frames in a shared dictionary
        with self.lock:
            self.rectified_frames[cam_id1] = rectified_left
            self.rectified_frames[cam_id2] = rectified_right
