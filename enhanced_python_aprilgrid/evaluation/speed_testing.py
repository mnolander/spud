import numpy as np
import cv2
from aprilgrid import Detector
from glob import glob
import time  # Import the time module

if __name__ == '__main__':    
    file_list = sorted(glob("C:/Users/Toby/Desktop/CapstoneProject/enhanced_python_aprilgrid/data/image_data\initial_png_single_camera/*.png"))
    detector = Detector('t16h5b1')
    detection_times = []  # List to store detection times

    for i, file_name in enumerate(file_list):
        # Load and preprocess image
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

        # Time only the detection process
        start_time = time.time()
        detections = detector.detect(img)
        detection_time = time.time() - start_time

        detection_times.append(detection_time)  # Store the detection time

        # Debug: Confirm detection time without other operations
        print(f"Frame {i}: Detected {len(detections)} tags in {detection_time:.3f} seconds")

    # Calculate the average detection time
    avg_detection_time = sum(detection_times) / len(detection_times)

    # Print summary
    print("\nDetection times per frame:")
    for i, t in enumerate(detection_times):
        print(f"Frame {i}: {t:.3f} seconds")

    print(f"\nTotal frames: {len(file_list)}")
    print(f"Average detection time: {avg_detection_time:.3f} seconds per frame")
for detection in detections:
            tag_id = detection.tag_id
            corners = detection.corners

            # Convert corners to a proper numpy array
            corners = np.array(corners)

            # Create a color image to overlay the results (BGR format)
            output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Draw the corners in different colors
            for i, corner in enumerate(corners):
                corner = corner.flatten().astype(int)  # Flatten and ensure the corner is an integer tuple
                print(f"Corner {i}: {corner}")  # Check the corner format
                color = (0, 0, 255) if i == 0 else (0, 255, 0) if i == 1 else (255, 0, 0) if i == 2 else (0, 255, 255)
                # Draw a circle at each corner
                cv2.circle(output_img, tuple(corner), 5, color, -1)

            # Draw lines connecting the corners
            for i in range(4):
                start = tuple(corners[i].flatten().astype(int))  # Flatten corner before using
                end = tuple(corners[(i + 1) % 4].flatten().astype(int))  # Connect corners in a loop
                cv2.line(output_img, start, end, (255, 255, 255), 2)

            # Put the detection ID in the center of the corners
            center = np.mean(corners, axis=0).flatten().astype(int)
            cv2.putText(output_img, str(tag_id), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Resize the output image to make it smaller for easier viewing
            output_img_resized = cv2.resize(output_img, (2012, 1518))

            # Show the result
            cv2.imshow(f"Detection {tag_id}", output_img_resized)

            # Wait for a key press to continue to the next image (optional)
            cv2.waitKey(0)  # Press any key to move to the next image

    # Close all windows when done
cv2.destroyAllWindows()