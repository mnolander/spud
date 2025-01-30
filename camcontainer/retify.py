import cv2
import numpy as np
import threading

class StereoRectifier:
    def __init__(self):
        # Intrinsic parameters for both cameras
        self.intrinsics_cam0 = np.array([3431.6333053510584, 3425.4233661235844, 1037.0631103881203, 814.2748160563247])
        self.intrinsics_cam1 = np.array([3450.443379487295, 3442.6715206818894, 1000.8126349532237, 768.292008152334])

        # Transformation matrix between cameras
        self.T_cn_cnm1 = np.array([
            [0.9954706102064841, -0.002938725639798721, 0.09502435533453286, -0.1898177118318758],
            [0.002702420629833145, 0.999992928337077, 0.0026153773296546125, -0.001517588623992596],
            [-0.09503136923073889, -0.002346735488078846, 0.9954715122466737, 0.01757994327060201],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # Distortion coefficients for both cameras
        self.distortion_coeffs_cam0 = np.array([-0.29763392127051924, 0.8835065755916732, -0.0023637256530460084, -0.0007121489154046612])
        self.distortion_coeffs_cam1 = np.array([-0.2760132481960459, 0.7026370095608109, -0.0015314823046884355, -0.001072650889485783])

        # Extract rotation and translation
        R = self.T_cn_cnm1[:3, :3]  # Rotation matrix (3x3)
        T = self.T_cn_cnm1[:3, 3]   # Translation vector (3x1)

        # Intrinsic camera matrices
        K1 = np.array([
            [self.intrinsics_cam0[0], 0, self.intrinsics_cam0[2]],
            [0, self.intrinsics_cam0[1], self.intrinsics_cam0[3]],
            [0, 0, 1]
        ])

        K2 = np.array([
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
            K1, dist_coeffs1, K2, dist_coeffs2, self.image_size, R, T
        )

        # Rectification maps
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(K1, dist_coeffs1, self.R1_rect, self.P1, self.image_size, cv2.CV_32F)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(K2, dist_coeffs2, self.R2_rect, self.P2, self.image_size, cv2.CV_32F)
        
        print("self.P1")
        print(self.P1)
        print("self.P2")
        print(self.P2)

    def rectify_images(self, image_path_cam0, image_path_cam1):
        # Load images
        image_cam0 = cv2.imread(image_path_cam0)
        image_cam1 = cv2.imread(image_path_cam1)
        
        # bin the image to 2012, 1518 from the 4k one
        bin_factor = 2
        height, width = image_cam0.shape[:2]
        height, width = image_cam1.shape[:2]
        
        image_cam0 = cv2.resize(image_cam0, (width // bin_factor, height // bin_factor), interpolation=cv2.INTER_AREA)
        image_cam1 = cv2.resize(image_cam1, (width // bin_factor, height // bin_factor), interpolation=cv2.INTER_AREA)

        if image_cam0 is None or image_cam1 is None:
            print("Error: Could not load one or both images.")
            return

        # Apply rectification
        rectified_cam0 = cv2.remap(image_cam0, self.map1x, self.map1y, interpolation=cv2.INTER_LINEAR)
        rectified_cam1 = cv2.remap(image_cam1, self.map2x, self.map2y, interpolation=cv2.INTER_LINEAR)

        # Display images
        cv2.imshow('Rectified Camera 0', rectified_cam0)
        cv2.imshow('Rectified Camera 1', rectified_cam1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save images
        cv2.imwrite('rectified_left2.jpg', rectified_cam0)
        cv2.imwrite('rectified_right2.jpg', rectified_cam1)

        print("Rectified images saved successfully.")

# Example usage
if __name__ == "__main__":
    rectifier = StereoRectifier()
    rectifier.rectify_images('camera_DEV_1AB22C00E123_image_2.png', 'camera_DEV_1AB22C00E588_image_2.png')
