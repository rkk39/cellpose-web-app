import streamlit as st
import cv2
import numpy as np
import pandas as pd
from cellpose import models
from PIL import Image
from io import BytesIO

# Constants
PIXEL_TO_MICROMETER = 100 / 478  # µm per pixel
MIN_UM_AREA = 50
MAX_SIZE = 1024

# Load Cellpose model
model = models.Cellpose(gpu=True, model_type='cyto')

st.title("Batch Cell Segmentation and Area Analysis")

uploaded_files = st.file_uploader("Upload PNG microscopy image(s)", accept_multiple_files=True, type=["png"])

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"🔬 Processing: {uploaded_file.name}")
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        # Resize if too large
        if max(gray.shape[:2]) > MAX_SIZE:
            scale = MAX_SIZE / max(gray.shape[:2])
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Run Cellpose
        masks, flows, _, _ = model.eval(gray, diameter=150, channels=[0, 0], batch_size=1)

        overlay_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        height, width = masks.shape
        areas = []

        for cell_id in np.unique(masks):
            if cell_id == 0:
                continue

            mask = (masks == cell_id).astype(np.uint8)
            area_px = np.sum(mask)
            area_um = area_px * (PIXEL_TO_MICROMETER ** 2)

            if area_um < MIN_UM_AREA:
                continue

            coords = np.column_stack(np.where(mask > 0))
            if (
                np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == height - 1) or
                np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == width - 1)
            ):
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_img, contours, -1, (0, 0, 255), 2)

            M = cv2.moments(mask)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(overlay_img, str(cell_id), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)

            areas.append((cell_id, area_px, area_um))

        # Results table
        df = pd.DataFrame(areas, columns=["Cell ID", "Area (pixels^2)", "Area (µm^2)"])
        if not df.empty:
            avg_area = df["Area (µm^2)"].mean()
            df.loc["Average"] = ["—", "", avg_area]

        st.dataframe(df)

        # CSV Download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📄 Download Area Table (CSV)", csv, file_name=f"{uploaded_file.name}_areas.csv")

        # Labeled image preview
        st.image(overlay_img, caption="Labeled Mask Overlay", channels="BGR")

        # PNG Download
        is_success, buffer = cv2.imencode(".png", overlay_img)
        if is_success:
            st.download_button(
                label="🖼️ Download Masked Image",
                data=BytesIO(buffer),
                file_name=f"{uploaded_file.name}_mask_overlay.png",
                mime="image/png"
            )
