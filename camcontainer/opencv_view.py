import cv2 as cv
import numpy as np

# Load the images
image_path_cam0 = "camera_DEV_1AB22C00E588_image_3.png"
image_path_cam1 = "camera_DEV_1AB22C00E588_image_2.png"

img_cam0 = cv.imread(image_path_cam0)
img_cam1 = cv.imread(image_path_cam1)

if img_cam0 is None or img_cam1 is None:
    raise FileNotFoundError("Could not read one or both images")

# Intrinsic parameters (fx, fy, cx, cy) for both cameras
intrinsics_cam0 = np.array([3431.6333053510584, 3425.4233661235844, 1037.0631103881203, 814.2748160563247])
intrinsics_cam1 = np.array([3450.443379487295, 3442.6715206818894, 1000.8126349532237, 768.292008152334])

# Camera projection matrices (assuming translation along X)
P0 = np.array([[intrinsics_cam0[0], 0, intrinsics_cam0[2], 0],
               [0, intrinsics_cam0[1], intrinsics_cam0[3], 0],
               [0, 0, 1, 0]])

P1 = np.array([[intrinsics_cam1[0], 0, intrinsics_cam1[2], -100],  # Assume baseline of 100 units
               [0, intrinsics_cam1[1], intrinsics_cam1[3], 0],
               [0, 0, 1, 0]])

# Store clicked points for both cameras
clicked_points_cam0 = []
clicked_points_cam1 = []
current_camera = 0  # 0 for cam0, 1 for cam1

def mouse_events(event, x, y, flags, param):
    global clicked_points_cam0, clicked_points_cam1, current_camera

    if event == cv.EVENT_LBUTTONDOWN:
        if current_camera == 0:
            print(f"Cam0: Clicked at X={x}, Y={y}")
            clicked_points_cam0.append([x, y])
            current_camera = 1  # Switch to second camera
        else:
            print(f"Cam1: Clicked at X={x}, Y={y}")
            clicked_points_cam1.append([x, y])
            current_camera = 0  # Switch back to first camera

        if len(clicked_points_cam0) == len(clicked_points_cam1):  # Match pairs
            triangulate_pixels()

def triangulate_pixels():
    global clicked_points_cam0, clicked_points_cam1

    if len(clicked_points_cam0) != len(clicked_points_cam1):
        print("Need matching points from both cameras!")
        return

    # Convert to numpy arrays and make homogeneous coordinates
    pts0 = np.array(clicked_points_cam0, dtype=np.float32).T
    pts1 = np.array(clicked_points_cam1, dtype=np.float32).T

    pts0 = np.vstack((pts0, np.ones((1, pts0.shape[1]))))  # Homogeneous coordinates
    pts1 = np.vstack((pts1, np.ones((1, pts1.shape[1]))))

    # Triangulate points
    points_4D = cv.triangulatePoints(P0, P1, pts0[:2], pts1[:2])
    points_3D = points_4D[:3] / points_4D[3]  # Convert to 3D

    print(f"Triangulated 3D Points:\n{points_3D.T}")

    # Compute distances and print results
    for idx, point in enumerate(points_3D.T):
        x, y, z = point
        distance = np.linalg.norm([x, y, z])  # Euclidean distance from the camera
        print(f"Point {idx + 1}: 3D Position (X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}), Distance: {distance:.2f} units")

# Stack images horizontally for easy viewing
stacked_images = np.hstack((img_cam0, img_cam1))

cv.imshow("Stereo Image Viewer (Left: Cam0, Right: Cam1)", stacked_images)
cv.setMouseCallback("Stereo Image Viewer (Left: Cam0, Right: Cam1)", mouse_events)

cv.waitKey(0)
cv.destroyAllWindows()
