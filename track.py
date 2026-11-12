import cv2
import dlib
import numpy as np
import pyautogui
import pyautogui
from collections import deque

# Create a deque to store the last 9 mouse positions
mouse_positions = deque(maxlen=10)

# setup blob detector
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
blob_detector = cv2.SimpleBlobDetector_create(detector_params)

# setup pyautogui and find screen related coordinates
pyautogui.FAILSAFE = False
screen_width, screen_height = pyautogui.size()
screen_centre_x = screen_width//2
screen_centre_y = screen_height//2

calibrated = False
calibrated_eye_centre = None

# pupil detection using DIP techniques and cv2 built in blob detector
def DetectBlobs(img, detector):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_eq = cv2.equalizeHist(img_gray)
    _, img_thresh = cv2.threshold(img_eq, 35, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(img_thresh, None, iterations=2)
    img_dilate = cv2.dilate(img_erode, None, iterations=4)
    img_blur = cv2.medianBlur(img_dilate, 5)
    keypoints = detector.detect(img_blur)
    return keypoints

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat")

# Start the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

_, frame = cap.read()

# Get the dimensions of the image
frame_height, frame_width = frame.shape[:2]
frame_centre_x = frame_width//2
frame_centre_y = frame_height//2

# Calculate the position of the window on the screen
# (coordinates of the top-left corner of the window)
window_x = screen_centre_x - frame_centre_x
window_y = screen_centre_y - frame_centre_y

# Create the window with a name and move it to the centre
cv2.namedWindow('WebCam', cv2.WINDOW_NORMAL)
cv2.moveWindow('WebCam', window_x, window_y)

print("Instructions: \nFace must be well lit. \nHold head still. \nLook at the yellow circle for calibrations and press enter when ready to control the mouse.\nDo not move head afterwards.\n")

while True:

    # read frame from webcam and convert to gray
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # detect single face
    faces = detector(gray)
    if (len(faces) == 0): continue
    face = faces[0]

    # predict landmarks like eye, chin, nose, etc
    landmarks = predictor(image=gray, box=face)

    # we need a small image of the left eye
    left_eye_parts = [36, 37, 38, 39, 40, 41]

    # calculate coords of left eye rectangle
    min_x, min_y, max_x, max_y = None, None, None, None
    for part in left_eye_parts:
        coord = landmarks.part(part)
        if (min_x == None or coord.x < min_x): min_x = coord.x
        if (max_x == None or coord.x > max_x): max_x = coord.x
        if (min_y == None or coord.y < min_y): min_y = coord.y
        if (max_y == None or coord.y > max_y): max_y = coord.y

    # increase vertical range of image, otherwise important details missed
    min_y -= 10
    max_y += 10
    left_eye_img = frame[min_y:max_y, min_x:max_x]

    # detect pupil
    keypoints = DetectBlobs(left_eye_img, blob_detector)
    
    if (len(keypoints) == 0): continue

    eye_point = keypoints[0].pt # x and y of pupil relative to left_eye_img

    eye_frame_x = int(eye_point[0]) + min_x # x of pupil
    eye_frame_y = int(eye_point[1]) + min_y # y of pupil

    if (not calibrated):
        calibrated_eye_centre = (eye_frame_x, eye_frame_y)

        # show eye centre
        cv2.circle(img=frame, center=(eye_frame_x, eye_frame_y), radius=5, color=(0, 255, 0), thickness=1)
        
        # show screen centre
        cv2.circle(img=frame, center=(frame_width//2, frame_height//2), radius=5, color=(0, 255, 255), thickness=1)
        
        cv2.imshow("WebCam", frame)

        calibrated = cv2.waitKey(delay=1) == 13 # 13 is enter key
        if cv2.waitKey(delay=1) == ord('q'): break # if q pressed then exit
        continue

    # Calculate the distance of the pupil from the calibrated eye coordinates
    dx = eye_frame_x - calibrated_eye_centre[0]
    dy = eye_frame_y - calibrated_eye_centre[1]

    # scaling factors
    dx *= 110 
    dy *= 110

    # move mouse according to displacement
    new_mouse_x = screen_centre_x + dx
    new_mouse_y = screen_centre_y + dy

    # Add the new mouse position to the deque
    mouse_positions.append((new_mouse_x, new_mouse_y))

    # Calculate the average mouse position
    avg_mouse_x = sum(pos[0] for pos in mouse_positions) / len(mouse_positions)
    avg_mouse_y = sum(pos[1] for pos in mouse_positions) / len(mouse_positions)

    # Move the mouse cursor to the average position
    pyautogui.moveTo(avg_mouse_x, avg_mouse_y)

    # show eye circle
    cv2.circle(img=frame, center=(eye_frame_x, eye_frame_y), radius=5, color=(0, 0, 255), thickness=1)
    
    # show calibrated position
    cv2.circle(img=frame, center=(calibrated_eye_centre[0], calibrated_eye_centre[1]), radius=5, color=(0, 255, 0), thickness=1)
    
    cv2.imshow("WebCam", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(delay=1) == ord('q'): break

# When everything done, release the video capture
cap.release()

# Close all windows
cv2.destroyAllWindows()