import streamlit as st
from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
import json
import io

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Pothole Detection & Segmentation",
    page_icon="🕳️",
    layout="wide"
)

st.title("🕳️ AI Pothole Detection + Segmentation")
st.write("Bounding Boxes, Segmentation Overlay, and Binary Mask (Civil Ready)")

# ----------------------------
# Secure Roboflow Client
# ----------------------------
# IMPORTANT: Add your API key in Streamlit Cloud Secrets
# Settings → Secrets → Add:
# ROBOFLOW_API_KEY = "your_api_key_here"

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=st.secrets["ROBOFLOW_API_KEY"]
)

# ----------------------------
# Upload Image
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        # Read image
        image_bytes = uploaded_file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # Run Roboflow workflow
        with st.spinner("🔍 Running detection + segmentation..."):
            result = client.run_workflow(
                workspace_name="project1-mflte",
                workflow_id="detect-count-and-visualize-2",
                images={"image": image_bytes},  # ✅ send actual bytes
                use_cache=True
            )

        predictions = result[0]["predictions"]["predictions"]

        # =====================================================
        # 1️⃣ Bounding Boxes
        # =====================================================
        bbox_image = image.copy()

        for p in predictions:
            if all(k in p for k in ["x", "y", "width", "height"]):
                x1 = int(p["x"] - p["width"] / 2)
                y1 = int(p["y"] - p["height"] / 2)
                x2 = int(p["x"] + p["width"] / 2)
                y2 = int(p["y"] + p["height"] / 2)

                cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        bbox_image_rgb = cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB)

        # =====================================================
        # 2️⃣ Segmentation Overlay
        # =====================================================
        seg_overlay = image.copy()

        for obj in predictions:
            if "points" in obj:
                pts = np.array(
                    [[int(p["x"]), int(p["y"])] for p in obj["points"]],
                    dtype=np.int32
                )
                cv2.fillPoly(seg_overlay, [pts], (0, 255, 0))

        seg_overlay = cv2.addWeighted(image, 0.6, seg_overlay, 0.4, 0)
        seg_overlay_rgb = cv2.cvtColor(seg_overlay, cv2.COLOR_BGR2RGB)

        # =====================================================
        # 3️⃣ Binary Mask
        # =====================================================
        binary_mask = np.zeros((h, w), dtype=np.uint8)

        for obj in predictions:
            if "points" in obj:
                pts = np.array(
                    [[int(p["x"]), int(p["y"])] for p in obj["points"]],
                    dtype=np.int32
                )
                cv2.fillPoly(binary_mask, [pts], 255)

        # =====================================================
        # Display Results
        # =====================================================
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

        st.subheader("Binary Segmentation Mask (Civil Software Ready)")
        st.image(binary_mask, clamp=True)
        st.caption("White = pothole | Black = background")

        # =====================================================
        # Download Mask (In-Memory)
        # =====================================================
        mask_bytes = cv2.imencode(".png", binary_mask)[1].tobytes()

        st.download_button(
            label="⬇️ Download Binary Mask",
            data=mask_bytes,
            file_name="pothole_binary_mask.png",
            mime="image/png"
        )

        # Debug JSON
        with st.expander("Show raw predictions JSON"):
            st.code(json.dumps(result[0]["predictions"], indent=4), language="json")

    except Exception as e:
        st.error(f"Error occurred: {e}")
