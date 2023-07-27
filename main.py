import cv2
import mediapipe as mp
import time
import math
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import ctypes

# Constants for volume control
VOLUME_MIN = 0
VOLUME_MAX = 100
HAND_RANGE_MIN = 50
HAND_RANGE_MAX = 200

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize Video Stream
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Initialize time variables for FPS calculation
pTime = 0

# Function to set system volume using pycaw
def set_system_volume(volume_level):
    if volume_level < VOLUME_MIN:
        volume_level = VOLUME_MIN
    elif volume_level > VOLUME_MAX:
        volume_level = VOLUME_MAX

    # Get the default audio endpoint
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, 0, None)
    volume = ctypes.cast(interface, ctypes.POINTER(IAudioEndpointVolume))

    # Calculate the volume scalar (0.0 to 1.0)
    volume_scalar = volume_level / VOLUME_MAX

    # Set the system volume
    volume.SetMasterVolumeLevelScalar(volume_scalar, None)

def draw_volume_bar(frame, volume_level):
    volume_bar_height = int((volume_level / VOLUME_MAX) * frame.shape[0])
    volume_bar_color = (255, 0, 0)  # Blue color
    cv2.rectangle(frame, (frame.shape[1] - 50, frame.shape[0]), (frame.shape[1], frame.shape[0] - volume_bar_height), volume_bar_color, -1)

while True:
    # Read Frame
    ret, frame = cap.read()

    # Flip frame horizontally to display in mirrored mode
    frame = cv2.flip(frame, 1)

    # Convert Frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Make Detections
    results = hands.process(frame_rgb)

    # Check if Hands Detected
    if results.multi_hand_landmarks:
        # Loop Through Detected Hands
        for hand_landmarks in results.multi_hand_landmarks:
            # Loop Through Landmarks and Draw Them on Frame
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                # Draw contour line connecting landmarks
                if id > 0:
                    prev_x, prev_y = int(hand_landmarks.landmark[id-1].x * w), int(hand_landmarks.landmark[id-1].y * h)
                    cv2.line(frame, (prev_x, prev_y), (cx, cy), (255, 0, 0), 2)

            # Find the positions of the thumb tip and index finger tip
            thumb_tip_pos = None
            index_tip_pos = None
            for id, lm in enumerate(hand_landmarks.landmark):
                if id == 4:
                    thumb_tip_pos = (int(lm.x * w), int(lm.y * h))
                elif id == 8:
                    index_tip_pos = (int(lm.x * w), int(lm.y * h))

            # Draw a line between the thumb tip and index finger tip
            if thumb_tip_pos is not None and index_tip_pos is not None:
                cv2.line(frame, thumb_tip_pos, index_tip_pos, (0, 255, 0), 2)

                # Calculate the center of the line
                center_x = (thumb_tip_pos[0] + index_tip_pos[0]) // 2
                center_y = (thumb_tip_pos[1] + index_tip_pos[1]) // 2
                center = (center_x, center_y)

                # Calculate the length of the line (Euclidean distance)
                hand_range = math.sqrt((index_tip_pos[0] - thumb_tip_pos[0]) ** 2 + (index_tip_pos[1] - thumb_tip_pos[1]) ** 2)

                # Change the color of the center circle to yellow if the hand range is less than HAND_RANGE_MIN
                if hand_range < HAND_RANGE_MIN:
                    cv2.circle(frame, center, 5, (0, 255, 255), cv2.FILLED)
                    # Set volume to the minimum (0) if hand range is less than HAND_RANGE_MIN
                    volume_level = VOLUME_MIN
                else:
                    cv2.circle(frame, center, 5, (0, 0, 255), cv2.FILLED)
                    # Map the hand range to the volume range (VOLUME_MIN to VOLUME_MAX)
                    volume_level = int(np.interp(hand_range, [HAND_RANGE_MIN, HAND_RANGE_MAX], [VOLUME_MIN, VOLUME_MAX]))
                
                # Print the position of the center and the hand range in the console
                print(f'Center: ({center_x}, {center_y})')
                print(f'Hand Range: {hand_range:.2f} pixels')

                # Set the system volume
                set_system_volume(volume_level)

                # Draw the volume bar on the frame
                draw_volume_bar(frame, volume_level)

                # Display the volume level on the frame
                cv2.putText(frame, f'Volume: {volume_level}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)

    # Show Frame
    cv2.imshow("Hand Tracking", frame)

    # Exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release Video Stream and Destroy Windows
cap.release()
cv2.destroyAllWindows()
