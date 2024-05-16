import cv2
import numpy as np
import pickle

# Capture video from the camera
cap = cv2.VideoCapture(1)

write_flag = False
written_flag = False
parking_spaces = []  # Use a set to store unique parking space coordinates

while True:
    # Read frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Thresholding to extract white lines (parking spaces)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to dilate the white regions
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours of white lines
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw minimum enclosing rectangle around each block of parking spaces
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int64(box)
        cv2.drawContours(frame, [box], 0, (255, 255, 255), 50)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Thresholding to extract white lines (parking spaces)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological operations to dilate the white regions
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours of white lines
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw minimum enclosing rectangles around each segmented parking space
    for i, contour in enumerate(contours):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int64(box)

        if write_flag and not written_flag:
            start_point = (min(box[:, 0]), min(box[:, 1]))
            if start_point != (0, 0):
                end_point = (max(box[:, 0]), max(box[:, 1]))
                # Only append unique parking space coordinates
                parking_spaces.append((start_point, end_point))
            if i == len(contours) - 1:
                written_flag = True

        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
    # Display the frame
    cv2.imshow('Parking Space Detection', frame)

    key = cv2.waitKey(1) & 0xFF

    # Check if the 'y' key is pressed
    if key == ord('y'):
        write_flag = True

    if cv2.waitKey(1) & 0xFF == ord('q'):
        with open('pos', 'wb') as f:
            pickle.dump(parking_spaces, f)
        break

print("Parking space coordinates:")
for i, space in enumerate(parking_spaces):
    print(f"Parking Space {i + 1}: Start Point - {space[0]}, End Point - {space[1]}")

# Release the capture
cap.release()
cv2.destroyAllWindows()
