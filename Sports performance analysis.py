# # Sports performance analysis
import cv2
import numpy as np

# Load video file (replace with your path)
cap = cv2.VideoCapture('peoples2.mp4')

# Read the first frame
ret, first_frame = cap.read()
if not ret:
    print("Cannot read video file.")
    exit()

# Convert to grayscale and blur
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Define points to track using ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create mask to draw tracks
mask = np.zeros_like(first_frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow (i.e., track points)
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

    # Select good points
    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    # Draw tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        x_new, y_new = new.ravel()
        x_old, y_old = old.ravel()
        mask = cv2.line(mask, (int(x_new), int(y_new)), (int(x_old), int(y_old)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(x_new), int(y_new)), 5, (0, 0, 255), -1)

    # Overlay the mask on the frame
    output = cv2.add(frame, mask)
    cv2.imshow('Sports Performance Analysis - Player Movement', output)

    # Update for next frame
    prev_gray = gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()