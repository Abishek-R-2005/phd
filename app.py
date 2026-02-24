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
st.write("Bounding Boxes, Segmentation Overlay, Binary Mask, Area & Count")

uploaded_file = st.file_uploader("Upload Road Image", type=["jpg", "jpeg", "png"])

# ---------------------------------------------------
# ROBOFLOW CLIENT
# ---------------------------------------------------
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="YOUR_API_KEY"
)

WORKSPACE = "project1-mflte"
WORKFLOW_ID = "detect-count-and-visualize-2"


# ---------------------------------------------------
# PROCESS FRAME FUNCTION (UPDATED)
# ---------------------------------------------------
def process_frame(image, predictions):

    h, w, _ = image.shape
    total_image_area = h * w

    bbox_image = image.copy()
    seg_overlay = image.copy()
    binary_mask = np.zeros((h, w), dtype=np.uint8)

    pothole_count = 0
    pothole_areas = []
    total_damage_area = 0

    for p in predictions:

        # ---------------- BOUNDING BOX ----------------
        if all(k in p for k in ["x", "y", "width", "height"]):
            x1 = int(p["x"] - p["width"] / 2)
            y1 = int(p["y"] - p["height"] / 2)
            x2 = int(p["x"] + p["width"] / 2)
            y2 = int(p["y"] + p["height"] / 2)
            cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # ---------------- SEGMENTATION ----------------
        if "points" in p:

            pts = np.array(
                [[int(pt["x"]), int(pt["y"])] for pt in p["points"]],
                dtype=np.int32
            )

            # Fill overlay
            cv2.fillPoly(seg_overlay, [pts], (0, 255, 0))
            cv2.fillPoly(binary_mask, [pts], 255)

            # Calculate area (pixel area)
            area = cv2.contourArea(pts)

            pothole_count += 1
            pothole_areas.append(area)
            total_damage_area += area

            # Draw area text
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(
                bbox_image,
                f"{int(area)} px",
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    seg_overlay = cv2.addWeighted(image, 0.6, seg_overlay, 0.4, 0)

    damage_percent = (total_damage_area / total_image_area) * 100

    return (
        bbox_image,
        seg_overlay,
        binary_mask,
        pothole_count,
        pothole_areas,
        total_damage_area,
        damage_percent
    )


# ==========================================================
# IMAGE MODE
# ==========================================================
if uploaded_file:

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    image = cv2.imread(temp_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with st.spinner("Running detection + segmentation..."):
        result = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={"image": temp_path},
            use_cache=True
        )

    predictions = result[0]["predictions"]["predictions"]

    (
        bbox_image,
        seg_overlay,
        binary_mask,
        pothole_count,
        pothole_areas,
        total_damage_area,
        damage_percent
    ) = process_frame(image, predictions)

    # ---------------- DISPLAY RESULTS ----------------
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

    st.divider()

    # ---------------- METRICS ----------------
    st.subheader("📊 Pothole Analysis")

    colA, colB, colC = st.columns(3)

    colA.metric("Number of Potholes", pothole_count)
    colB.metric("Total Damage Area (px)", int(total_damage_area))
    colC.metric("Damage Percentage (%)", f"{damage_percent:.2f}")

    if pothole_count > 0:
        st.write("### Individual Pothole Areas (pixels)")
        for i, area in enumerate(pothole_areas, 1):
            st.write(f"Pothole {i}: {int(area)} px")

    os.remove(temp_path)
