import time
import cv2
import serial
import serial.tools.list_ports
from ultralytics import YOLO

# ---------- settings ----------
PORT_OVERRIDE = None       # set to "COM7" to force a port, e.g. PORT_OVERRIDE = "COM7"
CAMERA_INDEX = 0           # try 1 if the window is black
MIN_CONF = 0.70
STABLE_FRAMES = 5
REFRESH_S = 1.0

LABEL_TO_CMD = {
    "metal": "R", "glass": "R",
    "plastic": "P",
    "trash": "T",
    "paper": "G", "cardboard": "G",
}
LED_NAMES = {"R": "RED", "P": "WHITE 1", "T": "WHITE 2", "G": "GREEN"}
# ------------------------------


def find_arduino():
    if PORT_OVERRIDE:
        return PORT_OVERRIDE
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        text = ((p.description or "") + " " + (p.manufacturer or "")).lower()
        if any(k in text for k in ("arduino", "ch340", "usb-serial", "usb serial", "cp210", "ftdi")):
            return p.device
    candidates = [p.device for p in ports if p.device.upper() != "COM1"]
    return candidates[0] if len(candidates) == 1 else None


board = None
port = find_arduino()
if port is None:
    print("No Arduino found. Test mode: webcam works, no LEDs.")
else:
    try:
        board = serial.Serial(port, 9600, timeout=1)
        time.sleep(2)   # the Arduino restarts when the port opens
        print("Connected to", port)
    except serial.SerialException as e:
        print("Could not open", port, "-", e)
        print("Close the Arduino IDE / Serial Monitor and replug the USB.")
        print("Running in test mode: webcam works, no LEDs.")
        board = None

model = YOLO("best.pt")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

last_label, streak = None, 0
last_cmd, last_send = None, 0
status = "Show an item"

while True:
    ok, frame = cap.read()
    if not ok:
        print("Could not read from the camera.")
        break

    res = model(frame, verbose=False)[0]
    label = res.names[res.probs.top1].lower()
    conf = float(res.probs.top1conf)

    if conf >= MIN_CONF and label in LABEL_TO_CMD:
        streak = streak + 1 if label == last_label else 1
        last_label = label
    else:
        streak, last_label = 0, None

    now = time.time()
    if streak >= STABLE_FRAMES:
        cmd = LABEL_TO_CMD[label]
        if cmd != last_cmd or now - last_send > REFRESH_S:
            if board:
                board.write(cmd.encode())
            last_cmd, last_send = cmd, now
            status = f"{label} -> {LED_NAMES[cmd]} LED"
    elif last_cmd is not None and now - last_send > 3:
        status = "Show an item"
        last_cmd = None

    cv2.putText(frame, f"{label} {conf:.0%}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, status, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("Smart bin", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
if board:
    board.write(b"N")
    board.close()