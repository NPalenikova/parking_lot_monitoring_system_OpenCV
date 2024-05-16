import cv2
import pickle

# Global variables to track mouse events and rectangle drawing
drawing = False
start_x, start_y = -1, -1
end_x, end_y = -1, -1

try:
    with open('pos', 'rb') as f:
        spaces = pickle.load(f)

except FileNotFoundError:
    spaces = []


# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, end_x, end_y, spaces

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
        end_x, end_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_x, end_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        spaces.append(((start_x, start_y), (end_x, end_y)))

    elif event == cv2.EVENT_RBUTTONDOWN:
        for rect in spaces:
            if rect[0][0] < x < rect[1][0] and rect[0][1] < y < rect[1][1]:
                spaces.remove(rect)
                break


# Capture camera stream
cap = cv2.VideoCapture(1)

# Create a window and bind the mouse callback function
cv2.namedWindow("Camera Stream")
cv2.setMouseCallback("Camera Stream", mouse_callback)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.imread("busy_lot.png")

    # Draw existing rectangles
    for rect in spaces:
        cv2.rectangle(frame, rect[0], rect[1], (0, 255, 0), 2)

    # Draw the rectangle if left mouse button is pressed
    if drawing:
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Camera Stream", frame)
    cv2.setMouseCallback('Camera Stream', mouse_callback)

    # Check for key press and break the loop if 'q' is pressed
    if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty('Camera Stream', cv2.WND_PROP_VISIBLE) < 1):
        with open('pos', 'wb') as f:
            pickle.dump(spaces, f)
        break


# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
