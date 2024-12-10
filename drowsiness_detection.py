import cv2 as cv
import winsound  # For beep sound on Windows
import threading  # To play beep continuously in a separate thread

# Load Haar Cascade classifiers for face and eye detection
face_cascade = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("haarcascades/haarcascade_eye.xml")

# Check if the cascades are loaded successfully
if face_cascade.empty() or eye_cascade.empty():
    raise IOError("Error: Could not load Haar cascade files. Check the file paths.")

# Capture video from the webcam
video_capture = cv.VideoCapture(0)

# Check if the webcam is opened successfully
if not video_capture.isOpened():
    raise IOError("Error: Could not open webcam.")

print("Press 'q' to exit the video feed.")

# Variables to track eye closure
eye_closed_frames = 0  # Counter for consecutive frames with eyes closed
threshold = 10         # Number of frames to trigger the "Drowsy" warning
is_beeping = False     # Flag to manage the continuous beep sound

# Function to play a continuous beep sound in a separate thread
def play_beep():
    global is_beeping
    while is_beeping:
        winsound.Beep(1000, 500)  # Frequency = 1000 Hz, Duration = 500 ms

# Process the video feed frame by frame
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Flip the frame for a mirror-like effect
    frame = cv.flip(frame, 1)

    # Convert the frame to grayscale for detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    eyes_detected = False

    for (x, y, w, h) in faces:
        # Draw a rectangle around each detected face
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Define the region of interest (ROI) for eyes within the detected face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) > 0:
            eyes_detected = True  # Eyes are detected
            for (ex, ey, ew, eh) in eyes:
                # Draw a rectangle around each detected eye
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Check eye detection status
    if eyes_detected:
        eye_closed_frames = 0  # Reset the counter if eyes are detected
        if is_beeping:
            is_beeping = False  # Stop the beep sound
    else:
        eye_closed_frames += 1  # Increment the counter if eyes are not detected

    # Display warning and play beep sound if eyes are closed for too long
    if eye_closed_frames >= threshold:
        cv.putText(frame, "DROWSY!", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if not is_beeping:
            is_beeping = True
            threading.Thread(target=play_beep, daemon=True).start()  # Start the beep in a separate thread

    # Display the processed video feed
    cv.imshow('Video', frame)

    # Exit the loop when 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
video_capture.release()
cv.destroyAllWindows()
is_beeping = False  # Ensure the beep sound stops when the program exits
