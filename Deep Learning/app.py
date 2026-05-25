import cv2
from ultralytics import YOLO

# 1. Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# 2. Open your video source (0 for webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Get a list of all class IDs *except* 0 (person)
# COCO dataset has 80 classes (0 to 79)
non_person_classes = [i for i in range(1, 80)]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Run YOLO inference with limits:
    # 'classes' filters out people.
    # 'max_det=1' forces YOLO to only return the single highest-confidence object.
    results = model(
        frame, 
        classes=non_person_classes, 
        max_det=1, 
        verbose=False
    )

    # 4. Visualize the single detected object
    annotated_frame = results[0].plot()

    # 5. Display the output
    cv2.imshow("YOLO - Strict Single Object Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()