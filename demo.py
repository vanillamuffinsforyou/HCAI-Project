


import cv2
from ultralytics import YOLO

det_model = YOLO('yolov8n.pt')  

cls_model = YOLO('C:\\Users\\dhara\\Downloads\\best.pt')

class_names = ['Engaged', 'Not Engaged']

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not available")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = det_model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = det_model.names[cls_id]

        if label != 'person':
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = frame[y1:y2, x1:x2]

        resized = cv2.resize(cropped, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        cls_result = cls_model(rgb)
        pred_idx = cls_result[0].probs.top1
        cls_label = class_names[pred_idx]
        cls_conf = cls_result[0].probs.data[pred_idx].item()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = f"{cls_label} ({cls_conf*100:.1f}%)"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('YOLOv8 Detection + Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
