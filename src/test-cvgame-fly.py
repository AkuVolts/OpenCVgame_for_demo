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
fly_dx = random.choice([-1, 1]) * 10  # Horizontal movement direction and speed
fly_dy = random.choice([-1, 1]) * 10  # Vertical movement direction and speed

# Function to draw a circle (green target)
def draw_circle(image):
    global circle_x, circle_y
    cv2.circle(image, (circle_x, circle_y), 20, (0, 200, 0), 5)

# Function to draw the "fly" (red circle)
def draw_fly(image):
    global fly_x, fly_y
    cv2.circle(image, (fly_x, fly_y), 10, (255, 0, 0), -1)

# Function to move the fly randomly across the entire screen
def move_fly(image_width, image_height):
    global fly_x, fly_y, fly_dx, fly_dy
    fly_x += fly_dx
    fly_y += fly_dy

    # Change direction when the fly hits screen boundaries
    if fly_x <= 25 or fly_x >= image_width - 25:
        fly_dx *= -1
    if fly_y <= 25 or fly_y >= image_height - 25:
        fly_dy *= -1

# Function to increment score and create a new target
def increment_score(finger_x, finger_y):
    global score, circle_x, circle_y, fly_dx, fly_dy
    if (circle_x - 10) <= finger_x <= (circle_x + 10) and (circle_y - 10) <= finger_y <= (circle_y + 10):
        score += 1
        print("Score:", score)

        # Increase fly speed
        fly_dx += random.choice([-1, 1])
        fly_dy += random.choice([-1, 1])

        circle_x = random.randint(25, 615)
        circle_y = random.randint(25, 465)

# Function to check if the fly touches any part of the hand
def check_game_over(hand_landmarks, image_width, image_height):
    global fly_x, fly_y
    for landmark in hand_landmarks.landmark:
        pixel_coordinates = mp_drawing._normalized_to_pixel_coordinates(
            landmark.x, landmark.y, image_width, image_height
        )
        if pixel_coordinates:
            landmark_x, landmark_y = pixel_coordinates
            # Check if the fly is close to this landmark
            if (fly_x - 10) <= landmark_x <= (fly_x + 10) and (fly_y - 10) <= landmark_y <= (fly_y + 10):
                print("Game Over! Final Score:", score)
                return True
    return False

# Get the input from webcam
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

while cap.isOpened():
    _, frame = cap.read()
    if frame is None:
        print("No frame")
        break

    image = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
    image_height, image_width, _ = image.shape
    results = hands.process(image)

    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (0, 0, 200)
    cv2.putText(image, f"Score: {score}", (480, 30), text_font, 1, text_color, 4, cv2.LINE_AA)

    draw_circle(image)
    draw_fly(image)
    move_fly(image_width, image_height)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(50, 0, 150), thickness=4, circle_radius=3)
            )

            # Check if the fly touches any part of the hand
            if check_game_over(hand_landmarks, image_width, image_height):
                cap.release()
                cv2.destroyAllWindows()
                exit()

            # Increment score if the index finger touches the green circle
            landmark_at_point = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pixel_coordinates = mp_drawing._normalized_to_pixel_coordinates(
                landmark_at_point.x, landmark_at_point.y, image_width, image_height
            )
            if pixel_coordinates:
                finger_x, finger_y = pixel_coordinates
                cv2.circle(image, (finger_x, finger_y), 20, (0, 200, 0), 5)
                increment_score(finger_x, finger_y)

    cv2.imshow('Hand Tracking Game', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


