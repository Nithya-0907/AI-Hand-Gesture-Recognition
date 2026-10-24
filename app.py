
import cv2
import mediapipe as mp
import numpy as np
import joblib

MODEL_FILE = "models/gesture_model.pkl"

GESTURES = ["FIST", "OPEN_PALM", "PEACE", "THUMBS_UP"]


def main():
    try:
        model = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print("Model not found. Run python train_model.py first.")
        return

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("Could not open webcam.")
        return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        print("Live gesture recognition started. Press Q to quit.")

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            label = "No hand detected"

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                features = []
                for point in hand_landmarks.landmark:
                    features.extend([point.x, point.y, point.z])

                prediction = model.predict(
                    np.array(features).reshape(1, -1)
                )[0]

                label = f"Gesture: {prediction}"

            cv2.putText(
                frame, label, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0), 2
            )

            cv2.putText(
                frame, "Press Q to quit", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 0), 2
            )

            cv2.imshow("AI Hand Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
