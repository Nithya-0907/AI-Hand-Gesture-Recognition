
import cv2
import mediapipe as mp
import csv
import os

DATA_DIR = "data"
CSV_FILE = os.path.join(DATA_DIR, "gesture_data.csv")

GESTURES = ["OPEN_PALM", "FIST", "PEACE", "THUMBS_UP"]
SAMPLES_PER_GESTURE = 150

os.makedirs(DATA_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def collect():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    file_exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists or os.path.getsize(CSV_FILE) == 0:
            writer.writerow(
                ["label"] + [
                    f"{axis}{i}"
                    for i in range(21)
                    for axis in ("x", "y", "z")
                ]
            )

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        ) as hands:

            for gesture in GESTURES:
                count = 0
                print(f"\nNext gesture: {gesture}")
                print("Get ready. Press SPACE to start collecting samples.")
                started = False

                while count < SAMPLES_PER_GESTURE:
                    success, frame = camera.read()

                    if not success:
                        print("Could not read webcam frame.")
                        break

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    if results.multi_hand_landmarks:
                        landmarks = results.multi_hand_landmarks[0]
                        mp_draw.draw_landmarks(
                            frame, landmarks, mp_hands.HAND_CONNECTIONS
                        )

                    message = (
                        f"{gesture}: {count}/{SAMPLES_PER_GESTURE}"
                        if started else f"{gesture} - Press SPACE to start"
                    )

                    cv2.putText(
                        frame, message, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame, "Q = quit",
                        (15, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2
                    )
                    cv2.imshow("Gesture Data Collection", frame)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        camera.release()
                        cv2.destroyAllWindows()
                        print("Collection stopped.")
                        return

                    if key == 32:
                        started = True

                    if started and results.multi_hand_landmarks:
                        row = [gesture]

                        for point in results.multi_hand_landmarks[0].landmark:
                            row.extend([point.x, point.y, point.z])

                        writer.writerow(row)
                        file.flush()
                        count += 1

                if count < SAMPLES_PER_GESTURE:
                    print(f"Stopped early for {gesture}.")
                    break

    camera.release()
    cv2.destroyAllWindows()
    print(f"\nCollection finished. Dataset saved to {CSV_FILE}")


if __name__ == "__main__":
    collect()
