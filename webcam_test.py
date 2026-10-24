import cv2
from ultralytics import YOLO


model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    result = model(frame, verbose=False)[0]
    label = result.names[result.probs.top1]
    conf = float(result.probs.top1conf)
    cv2.putText(frame, f"{label} {conf:.2f}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Waste sorter", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()