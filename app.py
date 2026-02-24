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
    page_title="Pothole Segmentation Measurement System",
    page_icon="🕳️",
    layout="wide"
)

st.title("🕳️ Pothole Detection + Segmentation + Real Area (m²)")

uploaded_file = st.file_uploader("Upload Road Image", type=["jpg", "jpeg", "png"])

# ---------------------------------------------------
# SCALE CALIBRATION
# ---------------------------------------------------
st.sidebar.header("📏 Scale Calibration")

known_length_m = st.sidebar.number_input(
    "Known Object Length (meters)",
    min_value=0.01,
    value=1.0,
    step=0.1
)

pixel_length = st.sidebar.number_input(
    "That Object Length in Pixels",
    min_value=1,
    value=100,
    step=10
)

meter_per_pixel = known_length_m / pixel_length
area_conversion_factor = meter_per_pixel ** 2

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
# PROCESS FUNCTION
# ---------------------------------------------------
def process_frame(image, predictions):

    h, w, _ = image.shape

    bbox_image = image.copy()
    seg_overlay = image.copy()
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    pothole_count = 0
    pothole_real_areas = []

    for p in predictions:

        # ------------------ BOUNDING BOX DRAW ------------------
        if all(k in p for k in ["x", "y", "width", "height"]):
            x1 = int(p["x"] - p["width"] / 2)
            y1 = int(p["y"] - p["height"] / 2)
            x2 = int(p["x"] + p["width"] / 2)
            y2 = int(p["y"] + p["height"] / 2)

            cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # ------------------ SEGMENTATION AREA ------------------
        if "points" in p:

            pothole_count += 1

            mask = np.zeros((h, w), dtype=np.uint8)

            pts = np.array(
                [[int(pt["x"]), int(pt["y"])] for pt in p["points"]],
                dtype=np.int32
            )

            cv2.fillPoly(mask, [pts], 255)

            # Add to combined mask
            combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Pixel area (exact segmentation)
            pixel_area = np.sum(mask == 255)

            # Convert to m²
            real_area = pixel_area * area_conversion_factor
            pothole_real_areas.append(real_area)

            # Draw segmentation overlay
            cv2.fillPoly(seg_overlay, [pts], (0, 255, 0))

            # Centroid for labeling
            M = cv2.moments(mask)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = pts[0]

            cv2.putText(
                seg_overlay,
                f"{real_area:.2f} m2",
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    # Blend segmentation overlay
    seg_overlay = cv2.addWeighted(image, 0.6, seg_overlay, 0.4, 0)

    # Total area from combined mask (no double counting)
    total_pixels = np.sum(combined_mask == 255)
    total_damage_m2 = total_pixels * area_conversion_factor

    return (
        bbox_image,
        seg_overlay,
        combined_mask,
        pothole_count,
        pothole_real_areas,
        total_damage_m2
    )


# ---------------------------------------------------
# IMAGE MODE
# ---------------------------------------------------
if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    image = cv2.imread(temp_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with st.spinner("Running AI detection..."):
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
        pothole_real_areas,
        total_damage_m2
    ) = process_frame(image, predictions)

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    col1.image(image_rgb, caption="Original Image", use_container_width=True)
    col2.image(cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB),
               caption="Bounding Box Output",
               use_container_width=True)
    col3.image(cv2.cvtColor(seg_overlay, cv2.COLOR_BGR2RGB),
               caption="Segmentation Overlay (Area in m²)",
               use_container_width=True)
    col4.image(binary_mask,
               caption="Binary Segmentation Mask (Black & White)",
               clamp=True,
               use_container_width=True)

    st.divider()

    st.subheader("📊 Pothole Measurement Results")

    colA, colB = st.columns(2)
    colA.metric("Number of Potholes", pothole_count)
    colB.metric("Total Damaged Area (m²)", f"{total_damage_m2:.3f}")

    if pothole_count > 0:
        st.write("### Individual Pothole Areas (m²)")
        for i, area in enumerate(pothole_real_areas, 1):
            st.write(f"Pothole {i}: {area:.3f} m²")

    os.remove(temp_path)
