# Underwater Debris Detection & Analysis
![Project Banner](https://img.shields.io/badge/Project-Underwater%20Debris%20Detection-blue)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-green)
![Gradio Version](https://img.shields.io/badge/Gradio-4.x-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)

An interactive web application built with **Gradio** and **YOLOv8** for the detection and analysis of underwater trash and marine life. This tool can process both images and videos to identify debris, track unique items, and provide comprehensive data visualizations.

## Overview
This project provides a user-friendly interface and scripts for detecting, segmenting, and analyzing underwater debris (trash) in images and videos using deep learning models (YOLOv8, ONNX/PyTorch). It supports detailed analytics, severity scoring, and category breakdowns for detected trash.

## Features
- **Image Inference:** Upload or specify an image to detect and segment underwater trash, view annotated results, and download reports.
- **Video Inference:** Upload or specify a video to track and analyze trash across frames, download annotated videos and CSV logs, and view analytics plots.
- **Dashboard (if using app.py):** Visualize analytics including category distributions, per-frame trash counts, and severity levels.
- **Severity Scoring:** Automatically classifies the severity of detected trash based on count thresholds.
- **Category Bucketing:** Groups trash detections into meaningful categories (Plastic, Metal, Paper, etc.) for easier analysis.

## Technologies Used
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (ONNX & PyTorch)
- [Gradio](https://gradio.app/) (UI, if using app.py)
- OpenCV, Pillow, NumPy, Pandas, Matplotlib, ImageIO

## Installation
1. **Clone the repository**
   ```bash
   git clone (https://github.com/aunraza19/Underwater-debris-Detection.git)
   cd PythonProject
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download/Place Models**
   - Place your trained YOLOv8 models as `best.onnx` and/or `best.pt` in the project root.

## Usage
### 1. Gradio App (Recommended, if app.py is present)
Run the main app:
```bash
python app.py
```
- Open the provided Gradio link in your browser.
- Use the tabs to upload images/videos and view analytics.

### 2. Command-Line Scripts
#### Image Inference
```bash
python infere_image.py
```
- Outputs: Annotated image (`outputs/annotated.jpg`), JSON report (`outputs/report_image.json`).

#### Video Inference
```bash
python infere_video.py
```
- Outputs: Annotated video (`outputs/annotated_video.mp4`), per-frame CSV log (`outputs/video_log.csv`).

## File Structure
```
app.py                # Gradio UI and main logic (if present)
infere_image.py       # Standalone image inference script
infere_video.py       # Standalone video inference script
utils.py              # Helper functions (severity, bucketing)
requirements.txt      # Python dependencies
best.onnx, best.pt    # YOLOv8 models (place your own)
outputs/              # Results, logs, annotated media
```

## Customization
- **Class Names:** Edit `CLASS_NAMES` in scripts to match your dataset.
- **Severity Thresholds & Buckets:** Adjust in `utils.py` for custom scoring or categories.
- **Model:** Replace `best.onnx`/`best.pt` with your own trained models.

## Example Outputs
- **Annotated Images/Videos:** Segmented trash with bounding masks.
- **Pie Charts:** Trash category distribution (if using dashboard).
- **CSV Logs:** Per-frame detection counts and severity.
- **Dashboard:** Interactive analytics plots (bar, line, pie; if using app.py).

## Troubleshooting
- Ensure all dependencies are installed (`pip install -r requirements.txt`).
- Models must match the class order in `CLASS_NAMES`.
- For ONNX inference, use the `segment` task.
- If Gradio UI does not launch, check for errors in the terminal and verify model paths.

## License
This project is for research and educational purposes. Please cite the original YOLOv8 and Gradio repositories if used in publications.

## Credits
- YOLOv8 by Ultralytics
- Gradio by HuggingFace

---
For questions or issues, please open an issue or contact the maintainer.
