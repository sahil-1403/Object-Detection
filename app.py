"""
Streamlit Object Detection App
Uses pre-trained YOLOv8x model with COCO classes for real-time object detection on uploaded videos.
"""

import streamlit as st
import cv2
import tempfile
import os
from pathlib import Path
from ultralytics import YOLO
import time


# Page configuration
st.set_page_config(
    page_title="Object Detection App",
    page_icon="🎯",
    layout="wide"
)

# Title and description
st.title("🎯 Object Detection with YOLOv8x")
st.markdown("""
Upload a video file and detect objects using the pre-trained YOLOv8x model (COCO dataset).
The app will process your video and show detected objects with bounding boxes.
""")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Minimum confidence for detection. Lower values show more objects but may include false positives."
)

show_labels = st.sidebar.checkbox("Show Labels", value=True, help="Display class names and confidence scores")
show_fps = st.sidebar.checkbox("Show FPS", value=True, help="Display processing speed")


@st.cache_resource
def load_model():
    """Load the pre-trained YOLOv8x model (cached for performance)."""
    with st.spinner("Loading YOLOv8x model... (this happens once)"):
        model = YOLO('yolov8x.pt')  # Pre-trained on COCO dataset
    return model


def process_video(video_path, model, confidence, show_labels, show_fps):
    """
    Process video file and return path to annotated video.

    Args:
        video_path: Path to input video
        model: YOLO model instance
        confidence: Detection confidence threshold
        show_labels: Whether to show labels on detections
        show_fps: Whether to show FPS counter

    Returns:
        Path to processed video file
    """
    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Failed to open video file")
        return None

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create temporary output file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    frame_count = 0

    # Color palette for different classes
    colors = [
        (56, 56, 255), (255, 56, 56), (56, 255, 56), (0, 165, 255),
        (255, 0, 180), (200, 200, 0), (0, 255, 255), (180, 100, 255)
    ]

    try:
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            results = model(frame, conf=confidence, verbose=False)

            # Draw detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Get class and confidence
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls_id]

                    # Get color for this class
                    color = colors[cls_id % len(colors)]

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label if enabled
                    if show_labels:
                        label = f"{class_name} {conf:.2f}"

                        # Get text size for background
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 2
                        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

                        # Draw label background
                        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)

                        # Draw label text
                        cv2.putText(frame, label, (x1 + 2, y1 - 4), font, font_scale, (255, 255, 255), thickness)

            # Draw FPS if enabled
            if show_fps:
                elapsed_time = time.time() - start_time
                current_fps = (frame_count + 1) / elapsed_time if elapsed_time > 0 else 0
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Write frame
            out.write(frame)

            # Update progress
            frame_count += 1
            progress = frame_count / total_frames if total_frames > 0 else 0
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Processing: {frame_count}/{total_frames} frames ({progress*100:.1f}%)")

    finally:
        cap.release()
        out.release()

    progress_bar.progress(1.0)
    status_text.text(f"✅ Processing complete! Processed {frame_count} frames")

    return output_path


# Main app
def main():
    # Load model
    try:
        model = load_model()
        st.sidebar.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )

    if uploaded_file is not None:
        # Display video info
        file_details = {
            "Filename": uploaded_file.name,
            "File Size": f"{uploaded_file.size / (1024*1024):.2f} MB"
        }

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📹 Original Video")
            # Save uploaded file temporarily
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_input.write(uploaded_file.read())
            temp_input.close()

            # Display original video
            st.video(temp_input.name)

            # Show file details
            for key, value in file_details.items():
                st.text(f"{key}: {value}")

        with col2:
            st.subheader("🎯 Detected Objects")

            # Process button
            if st.button("🚀 Start Detection", type="primary", use_container_width=True):
                with st.spinner("Processing video..."):
                    output_path = process_video(
                        temp_input.name,
                        model,
                        confidence_threshold,
                        show_labels,
                        show_fps
                    )

                    if output_path and os.path.exists(output_path):
                        # Display processed video
                        st.video(output_path)

                        # Download button
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Download Annotated Video",
                                data=f.read(),
                                file_name=f"detected_{uploaded_file.name}",
                                mime="video/mp4",
                                use_container_width=True
                            )

                        # Clean up
                        os.unlink(output_path)
                    else:
                        st.error("Failed to process video")

                # Clean up input file
                os.unlink(temp_input.name)

    else:
        # Show instructions
        st.info("👆 Upload a video file to get started")

        # Show COCO classes
        with st.expander("📋 View Detectable Objects (COCO Classes)"):
            coco_classes = [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
            ]

            # Display in columns
            cols = st.columns(4)
            for idx, class_name in enumerate(coco_classes):
                col_idx = idx % 4
                cols[col_idx].write(f"• {class_name}")


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
This app uses YOLOv8x, a state-of-the-art object detection model pre-trained on the COCO dataset with 80 object classes.

**Features:**
- Real-time object detection
- Pre-trained on COCO dataset
- Adjustable confidence threshold
- Video download support
""")


if __name__ == "__main__":
    main()
