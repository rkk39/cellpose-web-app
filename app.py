import streamlit as st
import cv2
import numpy as np
import pandas as pd
from cellpose import models
from PIL import Image
from io import BytesIO

# Constants
PIXEL_TO_MICROMETER_40X = 0.2092
PIXEL_TO_MICROMETER_20X = 0.250
MAX_SIZE = 1024

# Load model once
model = models.Cellpose(gpu=True, model_type='cyto')

st.title("Cell Segmentation and Area Analysis")

# Step 1: User selects magnification
magnification = st.selectbox("Select Microscope Magnification:", ["40x", "20x"])

# Set config based on magnification
if magnification == "40x":
    diameter = 150
    pixel_to_micrometer = PIXEL_TO_MICROMETER_40X
    min_um_area = 50
    apply_area_filter = True
else:
    diameter = 50
    pixel_to_micrometer = PIXEL_TO_MICROMETER_20X
    min_um_area = 0
    apply_area_filter = False

uploaded_files = st.file_uploader("Upload microscopy image(s)", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)

master_data = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"Processing: {uploaded_file.name}")
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        if max(gray.shape[:2]) > MAX_SIZE:
            scale = MAX_SIZE / max(gray.shape[:2])
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        masks, _, _, _ = model.eval(gray, diameter=diameter, channels=[0, 0], batch_size=1)
        overlay_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        height, width = masks.shape
        areas = []

        for cell_id in np.unique(masks):
            if cell_id == 0:
                continue

            mask = (masks == cell_id).astype(np.uint8)
            area_px = np.sum(mask)
            area_um = area_px * (pixel_to_micrometer ** 2)

            if apply_area_filter and area_um < min_um_area:
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

        df = pd.DataFrame(areas, columns=["Cell ID", "Area (pixels^2)", "Area (µm^2)"])
        if not df.empty:
            avg_area = df["Area (µm^2)"].mean()
            df.loc["Average"] = ["—", "", avg_area]

        st.dataframe(df)
        df["Image"] = uploaded_file.name
        master_data.append(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📄 Download CSV", csv, file_name=f"{uploaded_file.name}_areas.csv")

        st.image(overlay_img, caption="Labeled Mask Overlay", channels="BGR")
        is_success, buffer = cv2.imencode(".png", overlay_img)
        if is_success:
            st.download_button(
                label="🖼️ Download Masked Image",
                data=BytesIO(buffer),
                file_name=f"{uploaded_file.name}_mask_overlay.png",
                mime="image/png"
            )

    # Final master Excel download
    if master_data:
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            for df in master_data:
                sheet_name = df["Image"].iloc[0].replace(" ", "_")[:31]
                df.drop(columns=["Image"]).to_excel(writer, sheet_name=sheet_name, index=False)
        st.download_button(
            "📘 Download All Results (Excel)",
            data=excel_buffer,
            file_name="master_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
