import streamlit as st
from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
import tempfile
import os

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Pothole Detection & Segmentation",
    page_icon="🕳️",
    layout="wide"
)

st.title("🕳️ Pothole Detection + Segmentation (Cloud Safe)")
st.write("Bounding Boxes, Segmentation Overlay, Binary Mask")

file_type = st.radio("Select Input Type", ["Image", "Video"])

if file_type == "Image":
    uploaded_file = st.file_uploader("Upload Road Image", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.file_uploader("Upload Road Video", type=["mp4", "mov", "avi", "mkv"])


# ---------------------------------------------------
# ROBOFLOW CLIENT
# ---------------------------------------------------
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="7l5BKkxbenEWpBCBPtSw"
)

WORKSPACE = "project1-mflte"
WORKFLOW_ID = "detect-count-and-visualize-2"


# ---------------------------------------------------
# PROCESS FRAME FUNCTION
# ---------------------------------------------------
def process_frame(image, predictions):
    h, w, _ = image.shape

    bbox_image = image.copy()
    seg_overlay = image.copy()
    binary_mask = np.zeros((h, w), dtype=np.uint8)

    for p in predictions:

        # Bounding Boxes
        if all(k in p for k in ["x", "y", "width", "height"]):
            x1 = int(p["x"] - p["width"] / 2)
            y1 = int(p["y"] - p["height"] / 2)
            x2 = int(p["x"] + p["width"] / 2)
            y2 = int(p["y"] + p["height"] / 2)
            cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Segmentation
        if "points" in p:
            pts = np.array([[int(pt["x"]), int(pt["y"])] for pt in p["points"]], dtype=np.int32)
            cv2.fillPoly(seg_overlay, [pts], (0, 255, 0))
            cv2.fillPoly(binary_mask, [pts], 255)

    seg_overlay = cv2.addWeighted(image, 0.6, seg_overlay, 0.4, 0)

    return bbox_image, seg_overlay, binary_mask


# ==========================================================
# IMAGE MODE
# ==========================================================
if uploaded_file and file_type == "Image":

    # Save to temp file (CLOUD SAFE FIX)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    image = cv2.imread(temp_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with st.spinner("Running detection..."):
        result = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={"image": temp_path},
            use_cache=True
        )

    # DEBUG (remove later if you want)
    # st.json(result)

    predictions = result[0]["predictions"]["predictions"]

    bbox_image, seg_overlay, binary_mask = process_frame(image, predictions)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original")
        st.image(image_rgb, use_container_width=True)

    with col2:
        st.subheader("Bounding Boxes")
        st.image(cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col3:
        st.subheader("Segmentation Overlay")
        st.image(cv2.cvtColor(seg_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.divider()
    st.subheader("Binary Mask")
    st.image(binary_mask, clamp=True)

    os.remove(temp_path)


# ==========================================================
# VIDEO MODE
# ==========================================================
if uploaded_file and file_type == "Video":

    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    cap = cv2.VideoCapture(video_path)

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.info(f"Input FPS: {input_fps:.2f}")
    st.warning("Processing 1 frame per second")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out_bbox = "bbox_output.mp4"
    writer_bbox = cv2.VideoWriter(out_bbox, fourcc, 1, (width, height))

    frame_count = 0
    skip_frames = int(input_fps)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)

    with st.spinner("Processing video..."):

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames == 0:

                temp_frame = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                cv2.imwrite(temp_frame.name, frame)

                result = client.run_workflow(
                    workspace_name=WORKSPACE,
                    workflow_id=WORKFLOW_ID,
                    images={"image": temp_frame.name},
                    use_cache=True
                )

                predictions = result[0]["predictions"]["predictions"]
                bbox_image, _, _ = process_frame(frame, predictions)

                writer_bbox.write(bbox_image)

                os.remove(temp_frame.name)

            frame_count += 1
            progress.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    writer_bbox.release()

    st.success("Video Processing Completed")
    st.video(out_bbox)

    os.remove(video_path)
