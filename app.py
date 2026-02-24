import streamlit as st
import numpy as np
import requests
import json
import io
from PIL import Image, ImageDraw

st.set_page_config(
    page_title="Pothole Detection & Segmentation",
    page_icon="🕳️",
    layout="wide"
)

st.title("🕳️ AI Pothole Detection + Segmentation")
st.write("Bounding Boxes, Segmentation Overlay, and Binary Mask (Cloud Safe Version)")

uploaded_file = st.file_uploader("Upload Road Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    w, h = image.size

    st.image(image, caption="Original Image", use_container_width=True)

    # Convert image to bytes
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()

    # Roboflow REST API
    api_key = st.secrets["ROBOFLOW_API_KEY"]

    url = f"https://serverless.roboflow.com/project1-mflte/detect-count-and-visualize-2?api_key={api_key}"

    with st.spinner("🔍 Running detection + segmentation..."):
        response = requests.post(
            url,
            files={"file": img_bytes}
        )

    if response.status_code != 200:
        st.error("Roboflow API Error")
        st.stop()

    result = response.json()
    predictions = result.get("predictions", [])

    # ==========================
    # Bounding Box Image
    # ==========================
    bbox_image = image.copy()
    draw_bbox = ImageDraw.Draw(bbox_image)

    for p in predictions:
        if all(k in p for k in ["x", "y", "width", "height"]):
            x1 = p["x"] - p["width"] / 2
            y1 = p["y"] - p["height"] / 2
            x2 = p["x"] + p["width"] / 2
            y2 = p["y"] + p["height"] / 2

            draw_bbox.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # ==========================
    # Segmentation Overlay
    # ==========================
    seg_overlay = image.copy()
    draw_seg = ImageDraw.Draw(seg_overlay, "RGBA")

    for obj in predictions:
        if "points" in obj:
            pts = [(p["x"], p["y"]) for p in obj["points"]]
            draw_seg.polygon(pts, fill=(0, 255, 0, 120))

    # ==========================
    # Binary Mask
    # ==========================
    binary_mask = Image.new("L", (w, h), 0)
    draw_mask = ImageDraw.Draw(binary_mask)

    for obj in predictions:
        if "points" in obj:
            pts = [(p["x"], p["y"]) for p in obj["points"]]
            draw_mask.polygon(pts, fill=255)

    # ==========================
    # Display Results
    # ==========================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Bounding Boxes")
        st.image(bbox_image, use_container_width=True)

    with col3:
        st.subheader("Segmentation Overlay")
        st.image(seg_overlay, use_container_width=True)

    st.divider()

    st.subheader("Binary Segmentation Mask")
    st.image(binary_mask)
    st.caption("White = pothole | Black = background")

    # Download mask
    mask_buffer = io.BytesIO()
    binary_mask.save(mask_buffer, format="PNG")

    st.download_button(
        "⬇️ Download Binary Mask",
        data=mask_buffer.getvalue(),
        file_name="pothole_binary_mask.png",
        mime="image/png"
    )

    with st.expander("Show Raw JSON"):
        st.code(json.dumps(result, indent=4), language="json")
