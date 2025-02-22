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
        intrinsics_cam0 = np.array([1620.4651055148497, 1618.625304097791, 488.1236111782669, 410.33669152243357])
        intrinsics_cam1 = np.array([1626.995844126752, 1624.6109433743875, 460.29127100375484, 361.9381172871196])

        # Transformation matrix between cameras
        extrinsics = np.array([
            [0.9950925698822555, -0.0044559731375204605, 0.09884797250587975, -0.18553604196639306],
            [0.002792804430802927, 0.9998523105371082, 0.016957516239548352, 0.0016774723406978195],
            [-0.09890893593875504, -0.016598235358041967, 0.9949580498566054, 0.020674624147283802],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # Distortion coefficients
        distortion_coeffs_cam0 = np.array([-0.2562170614950778, 0.5568550026732078, 0.001999253680834514, 8.506556525528789e-05])
        distortion_coeffs_cam1 = np.array([-0.268254529460207, 0.7099380328259791, -0.0012066896554370477, -0.006308679875068261])

        self.calibration = CameraCalibration(intrinsics_cam0, intrinsics_cam1, distortion_coeffs_cam0, distortion_coeffs_cam1, extrinsics)

        self.image_size = (1000, 758)

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
