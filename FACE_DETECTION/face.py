import cv2
from pathlib import Path


def load_face_cascade():
    cascade_path = Path(__file__).resolve().parents[1] / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        raise FileNotFoundError(f"Unable to load cascade file from {cascade_path}")
    return face_cascade


def main():
    face_cascade = load_face_cascade()
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        raise RuntimeError("Could not open the webcam. Please make sure your camera is connected and available.")

    while True:
        has_frame, img = webcam.read()
        if not has_frame or img is None:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.imshow("Face detection", img)
        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()