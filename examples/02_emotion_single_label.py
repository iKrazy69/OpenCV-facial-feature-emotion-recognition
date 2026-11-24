from deepface import DeepFace
import cv2
import time

# Open webcam
cap = cv2.VideoCapture(0)

# FPS calculation
prev_time = time.time()

print("Press 'S' to save a screenshot")
print("Press 'Q' to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to access webcam")
        break

    try:
        # Analyze emotion
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv"
        )

        # Get dominant emotion
        emotion = result[0]["dominant_emotion"]

        # Display emotion
        cv2.putText(
            frame,
            f"Emotion: {emotion}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    except Exception as e:
        print("Error:", e)

    # FPS Counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # Project title
    cv2.putText(
        frame,
        "Facial Expression Recognition",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("Emotion Recognition System", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save screenshot
    if key == ord('s'):
        filename = f"screenshot_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved as {filename}")

    # Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 