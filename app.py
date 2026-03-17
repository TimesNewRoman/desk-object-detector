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
    # Load pretrained YOLO model (will auto-download if not present)
    model = YOLO("yolov8n.pt")
    
    # Open webcam (0 = default camera)
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

        # Run YOLO inference on the frame
        results = model(frame, verbose=False)

        # Get first result (one frame → one result)
        result = results[0]

        # Store only boxes we care about
        filtered_boxes = []
        
        # Mapping from class_id → class_name
        names = result.names

        # Loop through detected boxes
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])     # numeric class id
                class_name = names[class_id]   # convert to name

                # Keep only allowed classes
                if class_name in ALLOWED_CLASSES:
                    filtered_boxes.append(box)

        # Copy original frame to draw on
        annotated_frame = frame.copy()

        # Draw filtered boxes manually
        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])    # bounding box coords
            confidence = float(box.conf[0])           # confidence score
            class_id = int(box.cls[0])
            class_name = names[class_id]

            label = f"{class_name} {confidence:.2f}"

            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        # Show the result frame
        cv2.imshow("Desk Object Detector", annotated_frame)
        
        # Press 'q' to exit loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    # Release resources          
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()