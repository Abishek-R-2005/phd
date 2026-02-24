import streamlit as st
from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
import tempfile
import os

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Pothole Detection & Segmentation",
    page_icon="🕳️",
    layout="wide"
)

st.title("🕳️ Pothole Detection + Segmentation (Image + Video)")
st.write("Bounding Boxes, Segmentation Overlay, and Binary Mask (Cloud Ready)")

file_type = st.radio("Select Input Type", ["Image", "Video"])

uploaded_file = None
if file_type == "Image":
    uploaded_file = st.file_uploader("Upload Road Image", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.file_uploader("Upload Road Video", type=["mp4", "mov", "avi", "mkv"])

# ----------------------------
# Secure Roboflow Client
# ----------------------------
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=st.secrets["ROBOFLOW_API_KEY"]
)

# ==========================================================
# Frame Processing Function
# ==========================================================
def process_frame(image, predictions):
    h, w, _ = image.shape

    # Bounding Boxes
    bbox_image = image.copy()
    for p in predictions:
        if all(k in p for k in ["x", "y", "width", "height"]):
            x1 = int(p["x"] - p["width"] / 2)
            y1 = int(p["y"] - p["height"] / 2)
            x2 = int(p["x"] + p["width"] / 2)
            y2 = int(p["y"] + p["height"] / 2)
            cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Segmentation Overlay
    seg_overlay = image.copy()
    for obj in predictions:
        if "points" in obj:
            pts = np.array([[int(p["x"]), int(p["y"])] for p in obj["points"]], dtype=np.int32)
            cv2.fillPoly(seg_overlay, [pts], (0, 255, 0))
    seg_overlay = cv2.addWeighted(image, 0.6, seg_overlay, 0.4, 0)

    # Binary Mask
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    for obj in predictions:
        if "points" in obj:
            pts = np.array([[int(p["x"]), int(p["y"])] for p in obj["points"]], dtype=np.int32)
            cv2.fillPoly(binary_mask, [pts], 255)

    binary_mask_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    return bbox_image, seg_overlay, binary_mask_bgr


# ==========================================================
# IMAGE MODE
# ==========================================================
if uploaded_file and file_type == "Image":

    image_bytes = uploaded_file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with st.spinner("Running detection + segmentation..."):
        result = client.run_workflow(
            workspace_name="project1-mflte",
            workflow_id="detect-count-and-visualize-2",
            images={"image": image_bytes},  # send bytes
            use_cache=True
        )

    predictions = result[0]["predictions"]["predictions"]

    bbox_image, seg_overlay, binary_mask_bgr = process_frame(image, predictions)

    bbox_image_rgb = cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB)
    seg_overlay_rgb = cv2.cvtColor(seg_overlay, cv2.COLOR_BGR2RGB)
    binary_mask_gray = cv2.cvtColor(binary_mask_bgr, cv2.COLOR_BGR2GRAY)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(image_rgb, use_container_width=True)

    with col2:
        st.subheader("Bounding Boxes")
        st.image(bbox_image_rgb, use_container_width=True)

    with col3:
        st.subheader("Segmentation Overlay")
        st.image(seg_overlay_rgb, use_container_width=True)

    st.divider()
    st.subheader("Binary Segmentation Mask")
    st.image(binary_mask_gray, clamp=True)
    st.caption("White = pothole | Black = background")


# ==========================================================
# VIDEO MODE
# ==========================================================
if uploaded_file and file_type == "Video":

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if input_fps == 0:
        st.error("Cannot read video FPS.")
        st.stop()

    st.info(f"🎥 Input Video FPS: {input_fps:.2f} | Resolution: {width}x{height}")
    st.warning("Processing 1 frame per second (Cloud Safe Mode)")

    skip_frames = int(input_fps)

    # Temp output files
    out_original = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out_bbox = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out_overlay = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out_mask = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer_original = cv2.VideoWriter(out_original, fourcc, 1, (width, height))
    writer_bbox = cv2.VideoWriter(out_bbox, fourcc, 1, (width, height))
    writer_overlay = cv2.VideoWriter(out_overlay, fourcc, 1, (width, height))
    writer_mask = cv2.VideoWriter(out_mask, fourcc, 1, (width, height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)

    with st.spinner("Processing video frames..."):

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames == 0:

                # Convert frame to bytes
                _, buffer = cv2.imencode(".jpg", frame)
                frame_bytes = buffer.tobytes()

                result = client.run_workflow(
                    workspace_name="project1-mflte",
                    workflow_id="detect-count-and-visualize-2",
                    images={"image": frame_bytes},
                    use_cache=True
                )

                predictions = result[0]["predictions"]["predictions"]

                bbox_image, seg_overlay, binary_mask_bgr = process_frame(frame, predictions)

                writer_original.write(frame)
                writer_bbox.write(bbox_image)
                writer_overlay.write(seg_overlay)
                writer_mask.write(binary_mask_bgr)

            frame_count += 1
            progress.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    writer_original.release()
    writer_bbox.release()
    writer_overlay.release()
    writer_mask.release()
    os.remove(tfile.name)

    st.success("✅ Video Processing Completed!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.video(out_original)
        st.subheader("Bounding Boxes")
        st.video(out_bbox)

    with col2:
        st.subheader("Segmentation Overlay")
        st.video(out_overlay)
        st.subheader("Binary Mask")
        st.video(out_mask)

    st.download_button("⬇️ Download Original", open(out_original, "rb"), file_name="original.mp4")
    st.download_button("⬇️ Download Bounding Box", open(out_bbox, "rb"), file_name="bbox.mp4")
    st.download_button("⬇️ Download Overlay", open(out_overlay, "rb"), file_name="overlay.mp4")
    st.download_button("⬇️ Download Mask", open(out_mask, "rb"), file_name="mask.mp4")
