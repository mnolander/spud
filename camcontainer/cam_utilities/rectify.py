import cv2
import numpy as np
import threading

class StereoRectifier:
    def __init__(self):
        # Intrinsic parameters for both cameras
        self.intrinsics_cam0 = np.array([3315.158744682164, 3317.4423656635427, 1044.9919174051424, 769.2791390484149])
        self.intrinsics_cam1 = np.array([3303.1303751889086, 3304.4918413538717, 1021.6391642423999, 738.8164474717523])

        # Transformation matrix between cameras
        self.T_cn_cnm1 = np.array([
            [0.9958785322848419, -0.001883418317043302, 0.09067745954553587, -0.19167885361868625],
            [0.0021215517599516157, 0.9999945494395888, -0.0025298437192514665, -0.0007459795575273124],
            [-0.09067220054856458, 0.0027117939739249486, 0.9958770999581846, 0.006433227010261678],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # Distortion coefficients for both cameras
        self.distortion_coeffs_cam0 = np.array([-0.26727533865087455, 0.6914547210589541, -0.00022631332732631944, 0.0006445742114670336])
        self.distortion_coeffs_cam1 = np.array([-0.25984247075537714, 0.609867968623631, 0.0008865986409872509, -0.000975406239237423])

        # Extract rotation and translation
        R = self.T_cn_cnm1[:3, :3]  # Rotation matrix (3x3)
        T = self.T_cn_cnm1[:3, 3]   # Translation vector (3x1)

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

        dist_coeffs1 = self.distortion_coeffs_cam0
        dist_coeffs2 = self.distortion_coeffs_cam1

        # Image size (width, height)
        self.image_size = (2012, 1518)

        # Stereo rectification
        self.R1_rect, self.R2_rect, self.P1, self.P2, self.Q, roi1, roi2 = cv2.stereoRectify(
            self.K1, dist_coeffs1, self.K2, dist_coeffs2, self.image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY
        )
        

        # Rectification maps
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(self.K1, dist_coeffs1, self.R1_rect, self.P1, self.image_size, cv2.CV_32F)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(self.K2, dist_coeffs2, self.R2_rect, self.P2, self.image_size, cv2.CV_32F)
        

    def rectify_images(self, image_path_cam0, image_path_cam1):
        # Load images
        image_cam0 = cv2.imread(image_path_cam0)
        image_cam1 = cv2.imread(image_path_cam1)
        
        # bin the image to 2012, 1518 as the image i took was not the right res
        bin_factor = 2
        height, width = image_cam0.shape[:2]
        height, width = image_cam1.shape[:2]
        
        print(height, width)
        
        image_cam0 = cv2.resize(image_cam0, (width // bin_factor, height // bin_factor), interpolation=cv2.INTER_AREA)
        image_cam1 = cv2.resize(image_cam1, (width // bin_factor, height // bin_factor), interpolation=cv2.INTER_AREA)

        if image_cam0 is None or image_cam1 is None:
            print("Error: Could not load one or both images.")
            return

        # undistorted_cam0 = cv2.undistort(image_cam0, self.K1, self.distortion_coeffs_cam0)
        # undistorted_cam1 = cv2.undistort(image_cam1, self.K2, self.distortion_coeffs_cam1)

        # Apply rectification
        rectified_cam0 = cv2.remap(image_cam0, self.map1x, self.map1y, interpolation=cv2.INTER_LINEAR)
        rectified_cam1 = cv2.remap(image_cam1, self.map2x, self.map2y, interpolation=cv2.INTER_LINEAR)

        # Display images
        cv2.imshow('Rectified Camera 0', rectified_cam0)
        cv2.imshow('Rectified Camera 1', rectified_cam1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save images
        cv2.imwrite('rectifiedT5_right6.jpg', rectified_cam0)
        cv2.imwrite('rectifiedT5_left6.jpg', rectified_cam1)

        print("Rectified images saved successfully.")

# Example usage
if __name__ == "__main__":
    rectifier = StereoRectifier()
    rectifier.rectify_images('camera_DEV_1AB22C00E123_imageT5_6.png', 'camera_DEV_1AB22C00E588_imageT5_6.png')
