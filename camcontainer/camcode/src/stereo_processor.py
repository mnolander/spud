import cv2
import numpy as np
import threading
import sys
import os

# If aprilgrid is under ../src:
cv2.ocl.setUseOpenCL(True)
cv2.setNumThreads(cv2.getNumberOfCPUs()) 

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
        intrinsics_cam0 = np.array([3283.252582900295, 3276.5576759347427, 1008.8577885733969, 788.8191633738176])
        intrinsics_cam1 = np.array([3285.0180948225093, 3279.112080228847, 996.8200004747581, 748.1774210103628])

        # Transformation matrix between cameras
        extrinsics = np.array([
            [0.9960097101961709, -0.0024348513556761214, 0.0892117071567023, -0.1935715151780031],
            [0.002301784889523529, 0.9999960797839735, 0.001594428739433365, -0.0014762468839194644],
            [-0.08921523962451598, -0.001382720347189578, 0.9960114101270022, 0.013836474915763909],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # Distortion coefficients
        distortion_coeffs_cam0 = np.array([-0.25185846070995066, 0.6871701552977757, -0.0002919388664898948, 0.0007789129793769401])
        distortion_coeffs_cam1 = np.array([-0.24257702122710484, 0.6622628024457236, 0.0008571551018062295, -0.0013174855531690249])

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
