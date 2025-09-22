# infer_video.py

from ultralytics import YOLO
import cv2, csv
from pathlib import Path
from utils import severity_from_count, bucket_class

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "best.onnx"
VIDEO_PATH = "manythings.mp4"  # replace with your video
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_VIDEO = OUTPUT_DIR / "annotated_video.mp4"
LOG_CSV = OUTPUT_DIR / "video_log.csv"

# -----------------------------
# Class names (for ONNX model)
# -----------------------------
CLASS_NAMES = ['rov', 'plant', 'animal_fish', 'animal_starfish', 'animal_shells',
               'animal_crab', 'animal_eel', 'animal_etc', 'trash_etc', 'trash_fabric',
               'trash_fishing_gear', 'trash_metal', 'trash_paper', 'trash_plastic',
               'trash_rubber', 'trash_wood']

# -----------------------------
# Load model
# -----------------------------
model = YOLO(MODEL_PATH, task='segment')

# -----------------------------
# Setup video reader/writer
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (w, h))

# -----------------------------
# Open CSV log
# -----------------------------
with open(LOG_CSV, "w", newline="") as f:
    csv_writer = csv.writer(f)
    header = ["frame_idx", "total_trash", "severity"] + CLASS_NAMES
    csv_writer.writerow(header)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.25, imgsz=640, save=False, verbose=False)
        r = results[0]

        counts_per_class = {name: 0 for name in CLASS_NAMES}
        bucket_counts = {}
        total_trash = 0

        for cls_id in r.boxes.cls.int().tolist():
            class_name = CLASS_NAMES[cls_id]
            bucket = bucket_class(class_name)
            if bucket:
                counts_per_class[class_name] += 1
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
                total_trash += 1

        severity = severity_from_count(total_trash)

        # Save CSV row
        row = [frame_idx, total_trash, severity] + [counts_per_class[name] for name in CLASS_NAMES]
        csv_writer.writerow(row)

        # Annotated frame
        annotated = r.plot()
        writer.write(annotated)

        frame_idx += 1

cap.release()
writer.release()
print(f"Annotated video saved at: {OUTPUT_VIDEO}")
print(f"Per-frame log saved at: {LOG_CSV}")
