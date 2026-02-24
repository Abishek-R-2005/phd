import streamlit as st
import numpy as np
import requests
import io
import tempfile
import os
import imageio
from PIL import Image, ImageDraw

st.set_page_config(
    page_title="Pothole Detection & Segmentation",
    page_icon="🕳️",
    layout="wide"
)

st.title("🕳️ Pothole Detection + Segmentation (Cloud Safe)")
st.write("Bounding Boxes, Segmentation Overlay, Binary Mask")

file_type = st.radio("Select Input Type", ["Image", "Video"])

uploaded_file = None
if file_type == "Image":
    uploaded_file = st.file_uploader("Upload Road Image", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.file_uploader("Upload Road Video", type=["mp4", "mov", "avi"])

API_KEY = st.secrets["ROBOFLOW_API_KEY"]
WORKFLOW_URL = f"https://serverless.roboflow.com/project1-mflte/detect-count-and-visualize-2?api_key={API_KEY}"

# ==========================================================
# Roboflow REST Call
# ==========================================================
def run_inference(image_bytes):
    response = requests.post(
        WORKFLOW_URL,
        files={"file": image_bytes}
    )
    return response.json()

# ==========================================================
# DRAW FUNCTION (Pillow)
# ==========================================================
def process_pil_image(image, predictions):

    w, h = image.size

    bbox_image = image.copy()
    overlay_image = image.copy()
    mask_image = Image.new("L", (w, h), 0)

    draw_bbox = ImageDraw.Draw(bbox_image)
    draw_overlay = ImageDraw.Draw(overlay_image, "RGBA")
    draw_mask = ImageDraw.Draw(mask_image)

    for p in predictions:

        if all(k in p for k in ["x", "y", "width", "height"]):
            x1 = p["x"] - p["width"] / 2
            y1 = p["y"] - p["height"] / 2
            x2 = p["x"] + p["width"] / 2
            y2 = p["y"] + p["height"] / 2
            draw_bbox.rectangle([x1, y1, x2, y2], outline="red", width=3)

        if "points" in p:
            pts = [(pt["x"], pt["y"]) for pt in p["points"]]
            draw_overlay.polygon(pts, fill=(0, 255, 0, 120))
            draw_mask.polygon(pts, fill=255)

    return bbox_image, overlay_image, mask_image

# ==========================================================
# IMAGE MODE
# ==========================================================
if uploaded_file and file_type == "Image":

    image = Image.open(uploaded_file).convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()

    with st.spinner("Running detection..."):
        result = run_inference(img_bytes)

    predictions = result.get("predictions", [])

    bbox, overlay, mask = process_pil_image(image, predictions)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Bounding Boxes")
        st.image(bbox, use_container_width=True)

    with col3:
        st.subheader("Segmentation Overlay")
        st.image(overlay, use_container_width=True)

    st.subheader("Binary Mask")
    st.image(mask)

    mask_buffer = io.BytesIO()
    mask.save(mask_buffer, format="PNG")

    st.download_button(
        "⬇️ Download Mask",
        data=mask_buffer.getvalue(),
        file_name="binary_mask.png",
        mime="image/png"
    )

# ==========================================================
# VIDEO MODE
# ==========================================================
if uploaded_file and file_type == "Video":

    temp_input = tempfile.NamedTemporaryFile(delete=False)
    temp_input.write(uploaded_file.read())
    temp_input.close()

    reader = imageio.get_reader(temp_input.name)
    fps = reader.get_meta_data()["fps"]

    st.info(f"Processing at 1 FPS (Input FPS: {fps})")

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer = imageio.get_writer(temp_output, fps=1)

    frame_count = 0

    with st.spinner("Processing video..."):

        for frame in reader:

            if frame_count % int(fps) == 0:

                pil_frame = Image.fromarray(frame).convert("RGB")

                buffer = io.BytesIO()
                pil_frame.save(buffer, format="JPEG")
                frame_bytes = buffer.getvalue()

                result = run_inference(frame_bytes)
                predictions = result.get("predictions", [])

                _, overlay, _ = process_pil_image(pil_frame, predictions)

                writer.append_data(np.array(overlay))

            frame_count += 1

    reader.close()
    writer.close()
    os.remove(temp_input.name)

    st.success("✅ Video Processing Completed")
    st.video(temp_output)

    st.download_button(
        "⬇️ Download Processed Video",
        open(temp_output, "rb"),
        file_name="processed_video.mp4"
    )
