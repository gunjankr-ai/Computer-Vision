import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

capture_hands = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
    )
)
screen_width, screen_height = pyautogui.size()
camera = cv2.VideoCapture(0)
x1 = y1 = x2 = y2 = 0
timestamp = 0
while True:
    _, image = camera.read()
    image_height, image_width, _ = image.shape
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    timestamp += 1
    output_hands = capture_hands.detect_for_video(rgb_image, timestamp)
    all_hands = output_hands.hand_landmarks
    if all_hands:
        for hand in all_hands:
            one_hand_landmarks = hand
            for id, lm in enumerate(one_hand_landmarks):
                x = int(lm.x * image_width)
                y = int(lm.y * image_height)
                # print(x, y)
                if id == 8:
                    mouse_x = int(screen_width / image_width * x)
                    mouse_y = int(screen_height / image_height * y)
                    cv2.circle(img=image, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)
                    pyautogui.moveTo(mouse_x, mouse_y)
                    x1, y1 = x, y
                if id == 4:
                    cv2.circle(img=image, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)
                    x2, y2 = x, y
        dist = y2 - y1
        print(dist)
        if dist < 30:
            pyautogui.click()
            print("Click")
    cv2.imshow("Mouse Control Using Hands", image)
    key = cv2.waitKey(100)
    if key == 27:
        break
camera.release()
cv2.destroyAllWindows()