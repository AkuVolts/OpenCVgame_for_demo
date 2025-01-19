import cv2
import numpy as np
import random
import time
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

score = 0
circle_x = random.randint(25, 615)
circle_y = random.randint(25, 465)
fly_x = random.randint(25, 615)
fly_y = random.randint(25, 465)
fly_speed = 5
last_score_time = time.time()  # To track the last scoring event

# Load your custom images
target_image = cv2.imread("target.png", cv2.IMREAD_UNCHANGED)  # Your target image (e.g., a green circle replacement)
fly_image = cv2.imread("fly.png", cv2.IMREAD_UNCHANGED)  # Your fly image

# Resize images to appropriate dimensions
target_image = cv2.resize(target_image, (40, 40))
fly_image = cv2.resize(fly_image, (40, 40))

# Function to overlay PNG with transparency onto another image
def overlay_image(background, overlay, x, y):
    h, w, _ = overlay.shape
    alpha_s = overlay[:, :, 3] / 255.0  # Extract alpha channel
    alpha_b = 1.0 - alpha_s

    for c in range(3):  # Loop through color channels
        background[y:y+h, x:x+w, c] = (
            alpha_s * overlay[:, :, c] + alpha_b * background[y:y+h, x:x+w, c]
        )

# Function to draw the custom target
def draw_target(image):
    overlay_image(image, target_image, circle_x - 20, circle_y - 20)

# Function to move and draw the fly
def move_and_draw_fly(image):
    global fly_x, fly_y, fly_speed
    # Randomize fly movement
    fly_x += random.choice([-fly_speed, 0, fly_speed])
    fly_y += random.choice([-fly_speed, 0, fly_speed])
    fly_x = max(25, min(fly_x, image.shape[1] - 25))  # Keep within boundaries
    fly_y = max(25, min(fly_y, image.shape[0] - 25))
    overlay_image(image, fly_image, fly_x - 20, fly_y - 20)

# Function to increment score
def increment_score(finger_x, finger_y):
    global score, circle_x, circle_y, last_score_time
    if (circle_x - 20) <= finger_x <= (circle_x + 20) and (circle_y - 20) <= finger_y <= (circle_y + 20):
        score += 1
        last_score_time = time.time()  # Update score time
        print("Score:", score)
        # Generate new target position
        circle_x = random.randint(25, 615)
        circle_y = random.randint(25, 465)

# Function to check collision with fly
def check_fly_collision(hand_landmarks, image_width, image_height):
    global fly_x, fly_y
    for landmark in hand_landmarks.landmark:
        pixel_coordinates = mp_drawing._normalized_to_pixel_coordinates(
            landmark.x, landmark.y, image_width, image_height
        )
        if pixel_coordinates:
            finger_x, finger_y = pixel_coordinates
            if (fly_x - 20) <= finger_x <= (fly_x + 20) and (fly_y - 20) <= finger_y <= (fly_y + 20):
                print("Game Over! The fly touched your hand.")
                return True
    return False

# Function to check timeout
def check_time_limit():
    global last_score_time
    if time.time() - last_score_time > 3:
        print("Game Over! You ran out of time.")
        return True
    return False

# Get webcam input
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

while cap.isOpened():
    _, frame = cap.read()
    if frame is None:
        print("No frame")
        break

    # Convert frame to RGB and flip
    image = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
    image_height, image_width, _ = image.shape
    results = hands.process(image)

    # Display score
    cv2.putText(image, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Draw target and move fly
    draw_target(image)
    move_and_draw_fly(image)

    # Check for hands and collisions
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(50, 0, 150), thickness=4, circle_radius=3)
            )
            # Check for collision with the fly
            if check_fly_collision(hand_landmarks, image_width, image_height):
                cap.release()
                cv2.destroyAllWindows()
                exit()
            # Check for index finger touching the target
            landmark_at_point = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pixel_coordinates = mp_drawing._normalized_to_pixel_coordinates(
                landmark_at_point.x, landmark_at_point.y, image_width, image_height
            )
            if pixel_coordinates:
                increment_score(pixel_coordinates[0], pixel_coordinates[1])

    # Check timeout
    if check_time_limit():
        cap.release()
        cv2.destroyAllWindows()
        break

    # Show the game frame
    cv2.imshow('Hand Tracking Game', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
