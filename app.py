import streamlit as st
from inference_sdk import InferenceHTTPClient
import google.generativeai as genai
import cv2
import numpy as np
import tempfile
import os

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Pothole Area + Depth + Volume",
    page_icon="🕳️",
    layout="wide"
)

st.title("🕳️ Pothole Detection + Depth + Volume Estimation")

# ---------------------------------------------------
# GEMINI API
# ---------------------------------------------------
GEMINI_API_KEY = "AIzaSyCWhZJOFnPgmkBmSQkqekodo08upi0TfR4"
genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------------------
# ROBOFLOW
# ---------------------------------------------------
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="7l5BKkxbenEWpBCBPtSw"
)

WORKSPACE = "project1-mflte"
WORKFLOW_ID = "detect-count-and-visualize-2"

# ---------------------------------------------------
# FILE
# ---------------------------------------------------
uploaded_file = st.file_uploader("Upload Road Image", type=["jpg", "jpeg", "png"])

# ---------------------------------------------------
# SCALE
# ---------------------------------------------------
st.sidebar.header("📏 Scale Calibration")

known_length_m = st.sidebar.number_input(
    "Known Object Length (meters)",
    min_value=0.01,
    value=1.0
)

pixel_length = st.sidebar.number_input(
    "Object Length in Pixels",
    min_value=1,
    value=100
)

meter_per_pixel = known_length_m / pixel_length
area_conversion_factor = meter_per_pixel ** 2


# ---------------------------------------------------
# GEMINI DEPTH ESTIMATION
# ---------------------------------------------------
def estimate_depth_with_gemini(image_path):
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = """
    Analyze this road pothole image.
    Estimate the pothole depth in meters.

    Return ONLY a single numeric value.
    Example:
    0.08
    """

    image_data = genai.upload_file(image_path)

    response = model.generate_content([prompt, image_data])

    try:
        depth = float(response.text.strip())
        return depth
    except:
        return 0.05  # fallback default 5 cm


# ---------------------------------------------------
# PROCESS
# ---------------------------------------------------
def process_frame(image, predictions, area_conversion_factor):
    h, w, _ = image.shape

    bbox_image = image.copy()
    seg_overlay = image.copy()
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    pothole_count = 0
    pothole_areas = []

    for p in predictions:
        if all(k in p for k in ["x", "y", "width", "height"]):
            x1 = int(p["x"] - p["width"] / 2)
            y1 = int(p["y"] - p["height"] / 2)
            x2 = int(p["x"] + p["width"] / 2)
            y2 = int(p["y"] + p["height"] / 2)

            cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if "points" in p:
            pothole_count += 1

            mask = np.zeros((h, w), dtype=np.uint8)

            pts = np.array(
                [[int(pt["x"]), int(pt["y"])] for pt in p["points"]],
                dtype=np.int32
            )

            cv2.fillPoly(mask, [pts], 255)

            combined_mask = cv2.bitwise_or(combined_mask, mask)

            pixel_area = np.sum(mask == 255)
            real_area = pixel_area * area_conversion_factor
            pothole_areas.append(real_area)

            cv2.fillPoly(seg_overlay, [pts], (0, 255, 0))

    seg_overlay = cv2.addWeighted(image, 0.6, seg_overlay, 0.4, 0)

    total_pixels = np.sum(combined_mask == 255)
    total_area = total_pixels * area_conversion_factor

    return bbox_image, seg_overlay, combined_mask, pothole_count, pothole_areas, total_area


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    image = cv2.imread(temp_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with st.spinner("Detecting pothole area..."):
        result = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={"image": temp_path},
            use_cache=True
        )

    predictions = result[0]["predictions"]["predictions"]

    bbox_image, seg_overlay, binary_mask, pothole_count, pothole_areas, total_area = process_frame(
        image,
        predictions,
        area_conversion_factor
    )

    with st.spinner("Estimating pothole depth using Gemini AI..."):
        estimated_depth = estimate_depth_with_gemini(temp_path)

    volume = total_area * estimated_depth

    # ---------------------------------------------------
    # DISPLAY
    # ---------------------------------------------------
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    c1.image(image_rgb, caption="Original", use_container_width=True)
    c2.image(cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB), caption="Bounding Box", use_container_width=True)
    c3.image(cv2.cvtColor(seg_overlay, cv2.COLOR_BGR2RGB), caption="Segmentation", use_container_width=True)
    c4.image(binary_mask, caption="Binary Mask", use_container_width=True)

    st.divider()

    st.subheader("📊 Final Measurement")

    m1, m2, m3 = st.columns(3)

    m1.metric("Potholes", pothole_count)
    m2.metric("Total Area", f"{total_area:.4f} m²")
    m3.metric("Estimated Depth", f"{estimated_depth:.4f} m")

    st.success(f"Estimated Volume = {volume:.5f} m³")

    st.write("### Individual Areas")
    for i, area in enumerate(pothole_areas, 1):
        individual_volume = area * estimated_depth
        st.write(
            f"Pothole {i}: Area = {area:.4f} m² | "
            f"Depth = {estimated_depth:.4f} m | "
            f"Volume = {individual_volume:.5f} m³"
        )

    os.remove(temp_path)
