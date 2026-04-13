"""
Streamlit Object Detection App
Uses pre-trained YOLOv8x model with COCO classes for real-time object detection on uploaded videos.
"""

import streamlit as st
import cv2
import tempfile
import os
import shutil
import subprocess
from pathlib import Path
from ultralytics import YOLO
import torch
import time


# COCO dataset classes
COCO_CLASSES = [
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

# Object class selection
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Object Filters")
filter_mode = st.sidebar.radio(
    "Filter Mode",
    ["Detect All Objects", "Detect Specific Objects Only"],
    help="Choose whether to detect all objects or only specific ones"
)

selected_classes = []
if filter_mode == "Detect Specific Objects Only":
    selected_classes = st.sidebar.multiselect(
        "Select Objects to Detect",
        options=COCO_CLASSES,
        default=["person", "cell phone", "laptop"],
        help="Only selected objects will be shown in the detected video"
    )


@st.cache_resource
def load_model():
    """Load the pre-trained YOLOv8x model (cached for performance)."""
    with st.spinner("Loading YOLOv8x model... (this happens once)"):
        # PyTorch 2.6+ defaults to weights_only=True, which can block older
        # Ultralytics checkpoints unless trusted classes are allowlisted.
        try:
            from ultralytics.nn.tasks import DetectionModel
            torch.serialization.add_safe_globals([DetectionModel])
        except Exception:
            pass
        model = YOLO('yolov8x.pt')  # Pre-trained on COCO dataset
    return model


def process_video(video_path, model, confidence, show_labels, show_fps, selected_classes=None):
    """
    Process video file and return path to annotated video.

    Args:
        video_path: Path to input video
        model: YOLO model instance
        confidence: Detection confidence threshold
        show_labels: Whether to show labels on detections
        show_fps: Whether to show FPS counter
        selected_classes: List of class names to filter (None means show all)

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
    if fps <= 0:
        fps = 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create temporary output files
    raw_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(raw_output_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        cap.release()
        st.error("Failed to initialize video writer")
        return None

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

                    # Filter by selected classes if specified
                    if selected_classes is not None and len(selected_classes) > 0:
                        if class_name not in selected_classes:
                            continue  # Skip this detection

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

    if frame_count == 0 or not os.path.exists(raw_output_path) or os.path.getsize(raw_output_path) == 0:
        if os.path.exists(raw_output_path):
            os.unlink(raw_output_path)
        if os.path.exists(output_path):
            os.unlink(output_path)
        st.error("No frames were written to output video")
        return None

    # Re-encode with H.264 for broad browser compatibility.
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        command = [
            ffmpeg_path,
            "-y",
            "-i",
            raw_output_path,
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            output_path,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            os.unlink(raw_output_path)
            progress_bar.progress(1.0)
            status_text.text(f"✅ Processing complete! Processed {frame_count} frames")
            return output_path

        # Fall back to raw OpenCV output if ffmpeg transcode fails.
        st.warning("FFmpeg re-encode failed; using raw output format.")
        if os.path.exists(output_path):
            os.unlink(output_path)
        progress_bar.progress(1.0)
        status_text.text(f"✅ Processing complete! Processed {frame_count} frames")
        return raw_output_path

    # FFmpeg not installed: return raw output and warn user.
    st.warning("FFmpeg not found. Install FFmpeg for better browser playback compatibility.")
    if os.path.exists(output_path):
        os.unlink(output_path)
    progress_bar.progress(1.0)
    status_text.text(f"✅ Processing complete! Processed {frame_count} frames")
    return raw_output_path


# Main app
def main():
    # Initialize session state
    if 'processed_video_path' not in st.session_state:
        st.session_state.processed_video_path = None
    if 'processed_filename' not in st.session_state:
        st.session_state.processed_filename = None

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
        # Reset processed video if a different file is uploaded
        if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
            # Clean up old processed video
            if 'processed_video_path' in st.session_state and st.session_state.processed_video_path:
                try:
                    os.unlink(st.session_state.processed_video_path)
                except:
                    pass
            st.session_state.processed_video_path = None
            st.session_state.processed_filename = None
            st.session_state.current_file = uploaded_file.name
            # Clean up old temp input file if it exists
            if 'temp_input_path' in st.session_state and st.session_state.temp_input_path:
                try:
                    os.unlink(st.session_state.temp_input_path)
                except:
                    pass
            # Create new temp file for the uploaded video
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_input.write(uploaded_file.read())
            temp_input.close()
            st.session_state.temp_input_path = temp_input.name

        # Display video info
        file_details = {
            "Filename": uploaded_file.name,
            "File Size": f"{uploaded_file.size / (1024*1024):.2f} MB"
        }

        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            with st.container(border=True):
                st.subheader("📹 Original Video")
                # Display original video using stored path
                st.video(st.session_state.temp_input_path)

                info_col1, info_col2 = st.columns([2, 1])
                with info_col1:
                    st.caption(f"Filename: `{file_details['Filename']}`")
                with info_col2:
                    st.caption(f"Size: `{file_details['File Size']}`")

        with col2:
            with st.container(border=True):
                st.subheader("🎯 Detected Objects")

                # Display processed video first so it aligns with original player.
                if 'processed_video_path' in st.session_state and st.session_state.processed_video_path:
                    if os.path.exists(st.session_state.processed_video_path):
                        st.video(st.session_state.processed_video_path)
                    else:
                        st.info("Processed video is no longer available. Please run detection again.")
                else:
                    st.info("Run detection to preview the annotated video here.")

                # Download button under player for consistent panel flow.
                if 'processed_video_path' in st.session_state and st.session_state.processed_video_path:
                    if os.path.exists(st.session_state.processed_video_path):
                        with open(st.session_state.processed_video_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Download Annotated Video",
                                data=f,
                                file_name=st.session_state.processed_filename,
                                mime="video/mp4",
                                use_container_width=True
                            )

                # Show filter info if active
                if filter_mode == "Detect Specific Objects Only" and selected_classes:
                    st.info(f"🔍 Filtering for: {', '.join(selected_classes)}")
                elif filter_mode == "Detect Specific Objects Only" and not selected_classes:
                    st.warning("⚠️ Please select at least one object class to detect")

                # Process button
                if st.button("🚀 Start Detection", type="primary", use_container_width=True):
                    with st.spinner("Processing video..."):
                        # Prepare selected classes (None if detecting all)
                        classes_to_detect = selected_classes if filter_mode == "Detect Specific Objects Only" else None

                        output_path = process_video(
                            st.session_state.temp_input_path,
                            model,
                            confidence_threshold,
                            show_labels,
                            show_fps,
                            classes_to_detect
                        )

                        if output_path and os.path.exists(output_path):
                            # Store the output path in session state (don't delete it yet)
                            st.session_state.processed_video_path = output_path
                            st.session_state.processed_filename = f"detected_{uploaded_file.name}"
                            st.rerun()
                        else:
                            st.error("Failed to process video")
                            st.session_state.processed_video_path = None

    else:
        # Show instructions
        st.info("👆 Upload a video file to get started")

        # Show COCO classes
        with st.expander("📋 View All Detectable Objects (80 COCO Classes)"):
            # Display in columns
            cols = st.columns(4)
            for idx, class_name in enumerate(COCO_CLASSES):
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
- Object class filtering
- Video download support
""")


if __name__ == "__main__":
    main()
