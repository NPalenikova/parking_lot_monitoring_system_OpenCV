import cv2
import numpy as np
import pickle

# Capture video from the camera
cap = cv2.VideoCapture(1)

write_flag = False
saved_spaces = []


while True:
    # Read frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    parking_spaces = []

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
        area = cv2.contourArea(contour)
        if 3000 < area < 20000:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            cv2.drawContours(dilated, [box], 0, (255, 255, 255), 20)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Filter to get only inner contours
    inner_contours = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] != -1]

    # Draw minimum enclosing rectangles around each segmented parking space
    for i, contour in enumerate(inner_contours):
        area = cv2.contourArea(contour)
        if 1000 < area < 7000:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

    if write_flag:
        saved_spaces = parking_spaces
        write_flag = False

    # Display the frame
    cv2.imshow('Parking Space Detection', frame)

    key = cv2.waitKey(1) & 0xFF

    # Check if the 'y' key is pressed
    if key == ord('y'):
        write_flag = True

    if (key == ord('q')) or (cv2.getWindowProperty('Parking Space Detection', cv2.WND_PROP_VISIBLE) < 1):
        with open('pos', 'wb') as f:
            pickle.dump(parking_spaces, f)
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
