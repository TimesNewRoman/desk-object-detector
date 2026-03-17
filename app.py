import cv2
from ultralytics import YOLO


ALLOWED_CLASSES = {
    "bottle",
    "cup",
    "cell phone",
    "mouse",
    "keyboard",
    "book",
}


def main():
    model = YOLO("yolov8n.pt")
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

        results = model(frame, verbose=False)

        result = results[0]

        filtered_boxes = []
        names = result.names

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = names[class_id]

                if class_name in ALLOWED_CLASSES:
                    filtered_boxes.append(box)

        annotated_frame = frame.copy()

        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = names[class_id]

            label = f"{class_name} {confidence:.2f}"

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Desk Object Detector", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()