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
last_score_time = time.time() # Time of last score increment





# make a function to put green circle on the screen one at a time, with no fill
def draw_circle(image):
    global circle_x, circle_y, score
    cv2.circle(image, (circle_x, circle_y), 20, (0, 200, 0), 5)

# make a function to increment score every time a circle midpoint is touched by the index finger
def increment_score(finger_x, finger_y, font, color, image):
    global score, circle_x, circle_y, last_score_time
    if (circle_x - 10) <= finger_x <= (circle_x + 10) and (circle_y - 10) <= finger_y <= (circle_y + 10):
        score += 1
        last_score_time = time.time()
        print("Score:", score)

        # randomly generate new circle coordinates
        circle_x = random.randint(25, 615)
        circle_y = random.randint(25, 465)
        print(circle_x, circle_y)

        # display the incremented score
        text = cv2.putText(image, str(score), (590, 30), font, 1, color, 4, cv2.LINE_AA)

        # draw a new circle
        draw_circle(image)

# function to check if the circle is caught within time limit
def check_time_limit():
    global last_score_time
    current_time = time.time()

    # print("ct:", current_time)
    # print("lst:", last_score_time)

    if score == 0:
        if current_time - last_score_time > 8:
            print("Game Over! You ran out of time.")
            return True
    elif score >= 10:
        if current_time - last_score_time > 2:
            print("Game Over! You ran out of time.")
            return True
    else:
        if current_time - last_score_time > 3:  # 3-second timeout
            print("Game Over! You ran out of time.")
            return True
    return False


# get the input from webcam
cap = cv2.VideoCapture(0)

# initialize the hand model
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

# format the video

while cap.isOpened():
    # read the frame from the webcam
    _, frame = cap.read()
    if frame is None:
        print("No frame")
        break

    # change bgr display to rgb and mirror the frame and store it in the image variable
    image = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)

    # get the dimensions of the frame
    image_height, image_width, _ = image.shape

    # get the results from the hand model
    results = hands.process(image)

    # draw a circle at a random coordinate
    # circle_x = random.randint(25, image_width - 25)
    # circle_y = random.randint(25, image_height - 25)

    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (0, 0, 200)
    text = cv2.putText(image, "Score: ", (480, 30), text_font, 1, text_color, 4, cv2.LINE_AA)
    text = cv2.putText(image, str(score), (590, 30), text_font, 1, text_color, 4, cv2.LINE_AA)

    draw_circle(image)



    # if there are results
    if results.multi_hand_landmarks:
        for temp, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # draw the landmarks on the frame
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(50, 0, 150), thickness=4, circle_radius=3))

    # call the actual game functions and increase the score
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for point in mp_hands.HandLandmark:
                landmark_at_point = hand_landmarks.landmark[point]
                pixel_coordinates = mp_drawing._normalized_to_pixel_coordinates(
                    landmark_at_point.x, landmark_at_point.y, image_width, image_height
                )

                if point == mp_hands.HandLandmark.INDEX_FINGER_TIP:
                    if pixel_coordinates:
                        try:
                            cv2.circle(image, (pixel_coordinates[0], pixel_coordinates[1]), 20, (0, 200, 0), 5)
                            increment_score(pixel_coordinates[0], pixel_coordinates[1], text_font, text_color, image)

                        except:
                            pass
    
    if check_time_limit():
        cap.release()
        cv2.destroyAllWindows()
        break
                        

    # change the rgb image to bgr and display it
    cv2.imshow('Hand Tracking', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()