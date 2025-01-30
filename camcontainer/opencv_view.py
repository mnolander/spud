import cv2 as cv

# Load the image
image_path = "camera_DEV_1AB22C00E588_image_3.png"
img = cv.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Could not read the image: {image_path}")

# Initial zoom scale
scale = 1.0
min_scale = 0.5  # Minimum zoom (50% of original size)
max_scale = 3.0  # Maximum zoom (300% of original size)

# Function to handle mouse events (click to get pixel location, scroll to zoom)
def mouse_events(event, x, y, flags, param):
    global scale, img

    if event == cv.EVENT_LBUTTONDOWN:  # Left mouse button click
        print(f"Clicked at: X={x}, Y={y}")

    elif event == cv.EVENT_MOUSEWHEEL:  # Scroll to zoom
        if flags > 0:  # Scroll up (zoom in)
            scale = min(scale * 1.2, max_scale)
        else:  # Scroll down (zoom out)
            scale = max(scale * 0.8, min_scale)

        # Resize the image based on zoom level
        height, width = img.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_LINEAR)

        cv.imshow("Image Viewer", resized_img)

# Create OpenCV window
cv.imshow("Image Viewer", img)
cv.setMouseCallback("Image Viewer", mouse_events)

cv.waitKey(0)
cv.destroyAllWindows()
