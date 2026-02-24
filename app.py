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
    page_title="Pothole Segmentation Measurement",
    page_icon="🕳️",
    layout="wide"
)

st.title("🕳️ Pothole Segmentation + Real Area (m²)")

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
    api_key="7l5BKkxbenEWpBCBPtSw"
)

WORKSPACE = "project1-mflte"
WORKFLOW_ID = "detect-count-and-visualize-2"


# ---------------------------------------------------
# SEGMENTATION AREA PROCESSING
# ---------------------------------------------------
def process_frame(image, predictions):

    h, w, _ = image.shape

    bbox_image = image.copy()
    seg_overlay = image.copy()

    pothole_count = 0
    pothole_real_areas = []
    total_damage_pixels = 0

    for p in predictions:

        if "points" in p:

            # Create empty mask for this pothole
            mask = np.zeros((h, w), dtype=np.uint8)

            pts = np.array(
                [[int(pt["x"]), int(pt["y"])] for pt in p["points"]],
                dtype=np.int32
            )

            # Fill mask for this pothole
            cv2.fillPoly(mask, [pts], 255)

            # Count exact segmented pixels
            pixel_area = np.sum(mask == 255)

            # Convert to real area
            real_area = pixel_area * area_conversion_factor

            pothole_count += 1
            pothole_real_areas.append(real_area)
            total_damage_pixels += pixel_area

            # Draw segmentation overlay
            cv2.fillPoly(seg_overlay, [pts], (0, 255, 0))

            # Compute centroid for labeling
            M = cv2.moments(mask)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = pts[0]

            # Label segmentation with REAL AREA
            cv2.putText(
                seg_overlay,
                f"{real_area:.2f} m2",
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    seg_overlay = cv2.addWeighted(image, 0.6, seg_overlay, 0.4, 0)

    total_damage_m2 = total_damage_pixels * area_conversion_factor

    return seg_overlay, pothole_count, pothole_real_areas, total_damage_m2


# ---------------------------------------------------
# IMAGE MODE
# ---------------------------------------------------
if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    image = cv2.imread(temp_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with st.spinner("Running AI segmentation..."):
        result = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={"image": temp_path},
            use_cache=True
        )

    predictions = result[0]["predictions"]["predictions"]

    (
        seg_overlay,
        pothole_count,
        pothole_real_areas,
        total_damage_m2
    ) = process_frame(image, predictions)

    col1, col2 = st.columns(2)

    col1.image(image_rgb, caption="Original Image", use_container_width=True)
    col2.image(cv2.cvtColor(seg_overlay, cv2.COLOR_BGR2RGB),
               caption="Segmentation Area (m²)",
               use_container_width=True)

    st.divider()

    st.subheader("📊 Segmentation-Based Measurement")

    colA, colB = st.columns(2)
    colA.metric("Number of Potholes", pothole_count)
    colB.metric("Total Damaged Area (m²)", f"{total_damage_m2:.3f}")

    if pothole_count > 0:
        st.write("### Individual Pothole Areas (m²)")
        for i, area in enumerate(pothole_real_areas, 1):
            st.write(f"Pothole {i}: {area:.3f} m²")

    os.remove(temp_path)
