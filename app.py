import cv2
from ultralytics import YOLO


# Only keep these object classes from YOLO detections
ALLOWED_CLASSES = {
    "bottle",
    "cup",
    "cell phone",
    "mouse",
    "keyboard",
    "book",
}


def main():
    # Load pretrained YOLO model
    model = YOLO("yolov8n.pt")

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        # Read one frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Run YOLO inference
        results = model(frame, verbose=False)
        result = results[0]

        filtered_boxes = []
        names = result.names

        # Store object counts here
        counts = {}

        # Filter detections and count allowed classes
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = names[class_id]

                if class_name in ALLOWED_CLASSES:
                    filtered_boxes.append(box)
                    counts[class_name] = counts.get(class_name, 0) + 1

        # Copy frame for drawing
        annotated_frame = frame.copy()

        # Draw filtered boxes
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

        # Draw live counts in top-left corner
        y_offset = 30
        cv2.putText(
            annotated_frame,
            "Detected objects:",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        for class_name, count in counts.items():
            y_offset += 30
            text = f"{class_name}: {count}"
            cv2.putText(
                annotated_frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Show result
        cv2.imshow("Desk Object Detector", annotated_frame)

        # Quit on q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()