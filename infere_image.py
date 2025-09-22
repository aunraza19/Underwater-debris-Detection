from ultralytics import YOLO
import cv2
import json
from pathlib import Path
from utils import severity_from_count, bucket_class

MODEL_PATH = "best.onnx"
IMAGE_PATH = "sample.jpg"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_IMAGE = OUTPUT_DIR / "annotated.jpg"
OUTPUT_JSON = OUTPUT_DIR / "report_image.json"

# Manually define your class names
CLASS_NAMES = ['rov', 'plant', 'animal_fish', 'animal_starfish', 'animal_shells',
               'animal_crab', 'animal_eel', 'animal_etc', 'trash_etc', 'trash_fabric',
               'trash_fishing_gear', 'trash_metal', 'trash_paper', 'trash_plastic',
               'trash_rubber', 'trash_wood']

# Load ONNX model
model = YOLO(MODEL_PATH, task='segment')

# Run inference
results = model.predict(source=IMAGE_PATH, conf=0.25, imgsz=640)
r = results[0]

# Counting trash objects
counts_per_class = {}
bucket_counts = {}
total_trash = 0

for cls_id in r.boxes.cls.int().tolist():
    class_name = CLASS_NAMES[cls_id]  # <-- use manual mapping for ONNX
    bucket = bucket_class(class_name)
    if bucket:
        counts_per_class[class_name] = counts_per_class.get(class_name, 0) + 1
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        total_trash += 1

# Severity
severity = severity_from_count(total_trash)

# Save annotated image
annotated = r.plot()
cv2.imwrite(str(OUTPUT_IMAGE), annotated)

# Save report
report = {
    "image": IMAGE_PATH,
    "total_trash_objects": total_trash,
    "per_class": counts_per_class,
    "per_category": bucket_counts,
    "severity": severity
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(report, f, indent=2)

print("Report:", json.dumps(report, indent=2))
print(f"Annotated image saved at: {OUTPUT_IMAGE}")
print(f"JSON report saved at: {OUTPUT_JSON}")
