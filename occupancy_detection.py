import cv2
import numpy as np
import pickle
import serial
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import csv

# Loading list of parking space coordinates from a pickle file
with open('pos', 'rb') as f:
    pos_list = pickle.load(f)

# Serial communication setup
SERIAL_PORT = "COM7"
ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)


# Callback function for the trackbar
def on_trackbar(val):
    global percentage
    percentage = val / 100.0


# Function to check the status of each parking space in the given frame
def check_parking_space(img):
    global frame, percentage
    free_spaces = 0

    # Looping through each parking space coordinate
    for pos in pos_list:
        # Extracting parking space coordinates
        x1, y1 = pos[0]
        x2, y2 = pos[1]

        # Cropping the image to get only the parking space area
        img_crop = img[y1:y2, x1:x2]
        count_non_zero = cv2.countNonZero(img_crop)
        total = img_crop.size

        # Determine if the percentage of non-zero pixels exceeds the threshold
        if count_non_zero > (percentage * total):
            color = (0, 0, 255)
        else:
            free_spaces += 1
            color = (0, 255, 0)

        # Drawing a rectangle around the parking space and displaying the count of non-zero and total pixels inside it
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, f"{count_non_zero}/{total}", (x1, y2 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)

    current_time = time.time()
    # Sending free_spaces data through UART and appending to the list every 3 seconds
    if current_time - check_parking_space.last_send_time >= 3:
        ser.write(f"{free_spaces}".encode())
        check_parking_space.last_send_time = current_time
        occupancy_data.append((current_time, free_spaces))

    # Displaying the total number of free parking spaces out of the total number of parking spaces
    cv2.putText(frame, f'{free_spaces} / {len(pos_list)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)


check_parking_space.last_send_time = time.time()

cap = cv2.VideoCapture(1)

# Getting the dimensions of the video frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

# Setting up the video writer to write the processed video to a file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 codec
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

# Initialize percentage threshold value
percentage = 30 / 100.0  # 30%

# Create a window
cv2.namedWindow('Occupancy detection')

# Create a trackbar for the percentage threshold
cv2.createTrackbar('Hranica', 'Occupancy detection', int(percentage * 100), 100, on_trackbar)

# Initialize the list to store occupancy data
occupancy_data = []

while 1:
    # Reading a frame from the video capture
    ret, frame = cap.read()

    # Converting the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blurring the grayscale frame using a Gaussian filter
    blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), 1)

    # Applying adaptive thresholding to the blurred frame to binarize it
    threshold_frame = cv2.adaptiveThreshold(blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 25, 16)

    # Applying median filtering to the thresholded frame to remove noise
    frame_median = cv2.medianBlur(threshold_frame, 5)

    # Dilating the filtered frame to fill in gaps in the parking space boundaries
    kernel = np.ones((5, 5), np.uint8)
    dilated_frame = cv2.dilate(frame_median, kernel, iterations=1)

    check_parking_space(dilated_frame)

    # Displaying the frame
    cv2.imshow('Occupancy detection', frame)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

    if cv2.getWindowProperty('Occupancy detection', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Extract the times and free spaces from the occupancy data
times, free_spaces = zip(*occupancy_data)
times = [datetime.fromtimestamp(t) for t in times]  # Convert timestamps to datetime objects

# Generate the filename based on the date and time of the first occupancy data point
start_time = times[0]
filename_png = f"occupancy_{start_time.strftime('%Y%m%d_%H%M%S')}.png"
filename_csv = f"occupancy_{start_time.strftime('%Y%m%d_%H%M%S')}.csv"

# Plot the occupancy data
plt.figure(figsize=(10, 5))
plt.plot(times, free_spaces, label='Voľné parkovacie miesta')
plt.xlabel('Čas')
plt.ylabel('Počet voľných miest')
plt.title('Obsadenosť parkovacích miest za čas')
plt.legend()
plt.grid(True)

# Format the x-axis to show time in hours, minutes, and seconds
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

# Rotate date labels for better readability
plt.gcf().autofmt_xdate()

# Add the date to the corner of the graph
plt.text(0.95, 0.99, start_time.strftime('%Y-%m-%d %H:%M:%S'),
         verticalalignment='top', horizontalalignment='right',
         transform=plt.gca().transAxes,
         color='gray', fontsize=10)

# Save the plot as an image with the generated filename
plt.savefig(filename_png)

# Show the plot
plt.show()

# Save the occupancy data to a CSV file
with open(filename_csv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Timestamp', 'Free Spaces'])
    for t, free_space in occupancy_data:
        csvwriter.writerow([datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S'), free_space])
