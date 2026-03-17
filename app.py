import cv2
from ultralytics import YOLO


def main():
    # Load a small pretrained YOLO model
    model = YOLO("yolov8n.pt")

    # Open default webcam (0 = first camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Run detection on the frame
        results = model(frame, verbose=False)

        # Draw detections on the frame
        annotated_frame = results[0].plot()

        # Show the frame
        cv2.imshow("Desk Object Detector", annotated_frame)

        # Quit when q is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()