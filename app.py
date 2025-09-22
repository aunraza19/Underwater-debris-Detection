import gradio as gr
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
import os
from tqdm import tqdm
import imageio

# ----------------------------
# Load models
# ----------------------------
onnx_model_path = "best.onnx"
pytorch_model_path = "best.pt"

onnx_model = YOLO(onnx_model_path, task="segment")
pytorch_model = YOLO(pytorch_model_path)

# ----------------------------
# Class names
# ----------------------------
CLASS_NAMES = [
    'rov', 'plant', 'animal_fish', 'animal_starfish', 'animal_shells', 'animal_crab',
    'animal_eel', 'animal_etc', 'trash_etc', 'trash_fabric', 'trash_fishing_gear',
    'trash_metal', 'trash_paper', 'trash_plastic', 'trash_rubber', 'trash_wood'
]

# ----------------------------
# Helper functions
# ----------------------------
def bucket_class(class_name):
    if "trash" in class_name:
        if "plastic" in class_name: return "Plastic"
        if "metal" in class_name: return "Metal"
        if "paper" in class_name: return "Paper"
        if "rubber" in class_name: return "Rubber"
        if "wood" in class_name: return "Wood"
        if "fabric" in class_name: return "Fabric"
        if "fishing_gear" in class_name: return "Fishing Gear"
        return "Other Trash"
    return None

def severity_from_count(count):
    if count <= 5:
        return "Low"
    elif count <= 15:
        return "Moderate"
    else:
        return "Critical"

def bytesio_to_pil(buf):
    buf.seek(0)
    return Image.open(buf).convert("RGB")

# ----------------------------
# Image Inference
# ----------------------------
def process_image(image):
    results = onnx_model.predict(source=image, conf=0.25, imgsz=640)
    r = results[0]
    # Convert annotated image to RGB once for consistent display/saving
    annotated_bgr = r.plot(masks=True)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    counts_per_class = {name: 0 for name in CLASS_NAMES}
    bucket_counts = {}
    total_trash = 0

    if r.masks is not None:
        for mask_idx, mask in enumerate(r.masks.data):
            cls_id = int(r.boxes.cls[mask_idx])
            cls_name = CLASS_NAMES[cls_id]
            bucket = bucket_class(cls_name)
            if bucket:
                counts_per_class[cls_name] += 1
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
                total_trash += 1

    severity = severity_from_count(total_trash)
    stats_text = f"Total Unique Trash Items: {total_trash}\nSeverity: {severity}"

    # Save annotated RGB image
    annotated_pil = Image.fromarray(annotated_rgb)
    temp_img_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    annotated_pil.save(temp_img_file.name)
    temp_img_file.close()

    fig, ax = plt.subplots()
    if bucket_counts:
        ax.pie(bucket_counts.values(), labels=bucket_counts.keys(),
               autopct='%1.1f%%', colors=plt.cm.tab20.colors)
        ax.set_title("Trash Category Distribution")
    else:
        ax.text(0.5, 0.5, 'No trash detected', ha='center', va='center', transform=ax.transAxes)

    buf = BytesIO()
    fig.savefig(buf, format='png')
    pie_chart_img = bytesio_to_pil(buf)
    plt.close(fig)

    # Return 4 values to match Image tab outputs
    return annotated_rgb, stats_text, pie_chart_img, temp_img_file.name

# ----------------------------
# Video Inference
# ----------------------------
def process_video(video_file):
    if video_file is None:
        return None, None, None, "Error: No video uploaded.", None, None, None, None, None

    video_path = Path(video_file.name)
    results = pytorch_model.track(source=str(video_path), persist=True, tracker="bytetrack.yaml", stream=True)

    annotated_frames = []
    csv_rows = []
    unique_trash_ids = set()
    frame_counter = 0

    for r in tqdm(results, desc="Processing video frames"):
        # Convert annotated frame to RGB for consistent handling
        annotated_bgr = r.plot(masks=True)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        # Safely extract class IDs for counting, independent of tracking IDs
        class_ids = r.boxes.cls.int().tolist() if (r.boxes is not None and getattr(r.boxes, "cls", None) is not None) else []

        # Track unique trash IDs only when tracker IDs are available
        if r.boxes.id is not None:
            track_ids = r.boxes.id.int().tolist()
            for i, track_id in enumerate(track_ids):
                if i < len(class_ids):
                    cls_name = CLASS_NAMES[class_ids[i]]
                    bucket = bucket_class(cls_name)
                    if bucket:
                        unique_trash_ids.add(track_id)

        annotated_frames.append(annotated_rgb)

        frame_counts = {name: 0 for name in CLASS_NAMES}
        for cls_id in class_ids:
            frame_counts[CLASS_NAMES[cls_id]] += 1

        csv_rows.append({
            "frame_idx": frame_counter,
            "total_detections_in_frame": len(r.boxes),
            **frame_counts
        })
        frame_counter += 1

    if not annotated_frames:
        return None, None, None, "No frames processed.", None, None, None, None, None

    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    cap.release()

    output_video_path = Path(tempfile.gettempdir()) / f"annotated_video_{os.path.basename(video_path)}"
    writer = imageio.get_writer(str(output_video_path), fps=fps, codec='libx264')
    for f in annotated_frames:
        # Frames are RGB already; write directly
        writer.append_data(f)
    writer.close()

    annotated_still_frame = annotated_frames[len(annotated_frames) // 2]
    # Frame is RGB; convert directly to PIL
    annotated_still_pil = Image.fromarray(annotated_still_frame)
    temp_img_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    annotated_still_pil.save(temp_img_file.name)
    temp_img_file.close()

    total_unique_trash = len(unique_trash_ids)
    severity = severity_from_count(total_unique_trash)
    stats_text = f"Total Unique Trash Items: {total_unique_trash}\nSeverity: {severity}"

    overall_bucket_counts = {}
    for row in csv_rows:
        for cls_name in CLASS_NAMES:
            if cls_name in row and row[cls_name] > 0 and bucket_class(cls_name):
                bucket = bucket_class(cls_name)
                overall_bucket_counts[bucket] = overall_bucket_counts.get(bucket, 0) + row[cls_name]

    bar_df = pd.DataFrame(list(overall_bucket_counts.items()), columns=['Category', 'Count'])

    fig, ax = plt.subplots()
    if overall_bucket_counts:
        ax.pie(overall_bucket_counts.values(), labels=overall_bucket_counts.keys(),
               autopct='%1.1f%%', colors=plt.cm.tab20.colors)
        ax.set_title("Overall Trash Distribution")
    else:
        ax.text(0.5, 0.5, 'No trash detected', ha='center', va='center', transform=ax.transAxes)

    buf = BytesIO()
    fig.savefig(buf, format='png')
    pie_chart_img = bytesio_to_pil(buf)
    plt.close(fig)

    df = pd.DataFrame(csv_rows)
    trash_columns = [col for col in df.columns if 'trash' in col]
    category_df_melted = df[['frame_idx'] + trash_columns].melt(id_vars=['frame_idx'], var_name='category', value_name='count')

    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp_csv.name, index=False)
    tmp_csv.close()
    csv_path = tmp_csv.name

    return (
        str(output_video_path),
        str(output_video_path),
        str(temp_img_file.name),
        stats_text,
        pie_chart_img,
        csv_path,
        bar_df,
        df[['frame_idx','total_detections_in_frame']],
        category_df_melted
    )

# ----------------------------
# Gradio Interface
# ----------------------------
with gr.Blocks(theme=gr.themes.Soft(), title="Underwater Debris Detection") as demo:
    gr.Markdown("# 🌊 Underwater Debris Detection & Analysis")
    gr.Markdown("Upload an image or a video to analyze underwater trash.")

    # Image Tab
    with gr.Tab("Image Inference"):
        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(type="filepath", label="Upload Image")
                img_btn = gr.Button("Run Inference", variant="primary")
            with gr.Column(scale=2):
                img_output = gr.Image(label="Annotated Image")
                img_download = gr.File(label="Download Annotated Image", interactive=False)
        with gr.Row():
            stats_output = gr.Textbox(label="Trash Stats", show_copy_button=True)
            pie_chart_output = gr.Image(label="Trash Category Pie Chart")

        img_btn.click(
            fn=process_image,
            inputs=img_input,
            outputs=[img_output, stats_output, pie_chart_output, img_download]
        )

    # Video Tab
    with gr.Tab("Video Inference"):
        with gr.Row():
            with gr.Column(scale=1):
                vid_input = gr.File(label="Upload Video (mp4)")
                vid_btn = gr.Button("Run Video Inference", variant="primary")
            with gr.Column(scale=2):
                vid_output = gr.Video(label="Annotated Video")
                vid_download = gr.File(label="Download Annotated Video", interactive=False)
                vid_still_frame_download = gr.File(label="Download Annotated Still Frame", interactive=False)

        with gr.Row():
            vid_stats_output = gr.Textbox(label="Trash Stats", show_copy_button=True)
            vid_pie_chart_output = gr.Image(label="Overall Trash Distribution Pie Chart")

        csv_download = gr.File(label="Download Full CSV Log")

    # Dashboard Tab
    with gr.Tab("Dashboard"):
        gr.Markdown("### Comprehensive Video Analytics")

        with gr.Row():
            with gr.Column(scale=1):
                vid_bar_plot = gr.BarPlot(
                    x="Category",
                    y="Count",
                    title="Total Trash Counts",
                    tooltip=["Category", "Count"],
                    interactive=True
                )
            with gr.Column(scale=1):
                total_trash_plot = gr.LinePlot(
                    x="frame_idx",
                    y="total_detections_in_frame",
                    title="Total Trash Detections Per Frame",
                    x_title="Frame Index",
                    y_title="Total Detections",
                    interactive=True
                )
        with gr.Row():
            category_plot = gr.LinePlot(
                x="frame_idx",
                y="count",
                color="category",
                title="Trash Categories Per Frame",
                x_title="Frame Index",
                y_title="Count",
                overlay_alpha=0.3,
                interactive=True
            )

        # Connect video processing directly to the dashboard plots
        vid_btn.click(
            fn=process_video,
            inputs=vid_input,
            outputs=[
                vid_output,
                vid_download,
                vid_still_frame_download,
                vid_stats_output,
                vid_pie_chart_output,
                csv_download,
                vid_bar_plot,
                total_trash_plot,
                category_plot
            ]
        )

demo.launch()
