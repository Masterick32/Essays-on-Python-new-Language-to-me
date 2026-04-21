from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    results = model(frame, stream=True, conf=0.5)

    for r in results:
        annotated_frame = r.plot()

    cv2.imshow("Reconocimiento de objetos", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()