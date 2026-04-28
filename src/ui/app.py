from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from allium_cepa_classifier import AlliumCepaConfig
from allium_cepa_classifier import AlliumCepaModel
from allium_cepa_classifier import AlliumCepaResult


# -----------------------------
# Font helper (anti-aliased)
# -----------------------------
def _safe_font(size: int = 18):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for p in candidates:
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


# -----------------------------
# Resize helpers
# -----------------------------
def resize_image_and_detections(
    image: Image.Image, detections: pd.DataFrame, target_w: int
):
    if image.width == target_w or target_w <= 0:
        return image, detections

    scale = target_w / image.width
    target_h = int(round(image.height * scale))

    resized = image.resize(
        (target_w, target_h), resample=Image.Resampling.LANCZOS
    )

    det = detections.copy()
    for c in ["x_min", "x_max"]:
        if c in det.columns:
            det[c] = (det[c].astype(float) * scale).round().astype(int)
    for c in ["y_min", "y_max"]:
        if c in det.columns:
            det[c] = (det[c].astype(float) * scale).round().astype(int)

    return resized, det


# -----------------------------
# Drawing helpers
# -----------------------------
def draw_annotated(
    image: Image.Image,
    detections: pd.DataFrame,
    mode: Literal["all", "mitosis", "not_mitosis"] = "all",
    font_size: int = 18,
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = _safe_font(font_size)

    if detections.empty or "mitosis" not in detections.columns:
        return annotated

    df = detections.copy()
    if mode == "mitosis":
        df = df[df["mitosis"] == True]
    elif mode == "not_mitosis":
        df = df[df["mitosis"] == False]

    for _, row in df.iterrows():
        x_min, y_min = int(row["x_min"]), int(row["y_min"])
        x_max, y_max = int(row["x_max"]), int(row["y_max"])

        cls_name = row.get("class_name", "cell")
        mito = row.get("mitosis", None)
        mito_txt = "mitosis" if mito is True else ("not" if mito is False else "")
        label = f"{cls_name} | {mito_txt}".strip(" |")

        # bounding box
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)],
            outline="red",
            width=2,
        )

        # text size
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        text_y = max(0, y_min - text_h - 2)

        # background
        draw.rectangle(
            [x_min, text_y, x_min + text_w + 4, text_y + text_h + 4],
            fill="red",
        )

        # text (with stroke for readability)
        draw.text(
            (x_min + 2, text_y + 2),
            label,
            fill="white",
            font=font,
            stroke_width=2,
            stroke_fill="black",
        )

    return annotated


# -----------------------------
# Summary
# -----------------------------
def compute_summary(detections: pd.DataFrame) -> dict:
    total = int(len(detections))
    if total == 0 or "mitosis" not in detections.columns:
        return {"total": 0, "mitotic": 0, "non_mitotic": 0, "mitotic_index": 0.0}

    mitotic = int((detections["mitosis"] == True).sum())
    non_mitotic = int((detections["mitosis"] == False).sum())
    mitotic_index = (mitotic / total) * 100.0

    return {
        "total": total,
        "mitotic": mitotic,
        "non_mitotic": non_mitotic,
        "mitotic_index": float(mitotic_index),
    }


# -----------------------------
# Model loader
# -----------------------------
@st.cache_resource
def load_model() -> AlliumCepaModel:
    cfg = AlliumCepaConfig.from_yaml("config.yaml")
    return AlliumCepaModel(cfg)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Allium cepa – Mitosis Detector",
    layout="wide",
)

st.title("Allium cepa – Cell Detection & Mitosis Index")

with st.sidebar:
    st.header("Options")

    anno_choice = st.radio(
        "Annotations",
        ["Off", "All cells", "Mitosis only", "Not in mitosis only"],
        index=1,
    )

    use_conf_filter = st.checkbox("Apply confidence threshold", value=True)
    conf_thr = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)

    display_w = st.slider("Display width (px)", 400, 1600, 1100, 50)

    show_table = st.checkbox("Show detections table", value=False)
    download_csv = st.checkbox("Enable CSV download", value=True)

st.write("Upload an image to run detection + mitosis classification.")

uploaded = st.file_uploader(
    "Upload image", type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded is None:
    st.stop()

image = Image.open(uploaded).convert("RGB")

model = load_model()
result: AlliumCepaResult = model.predict(image)

detections = result.detections.copy()

if use_conf_filter and "confidence" in detections.columns:
    detections = detections[detections["confidence"] >= conf_thr].reset_index(drop=True)

summary = compute_summary(detections)

mode_map = {
    "Off": None,
    "All cells": "all",
    "Mitosis only": "mitosis",
    "Not in mitosis only": "not_mitosis",
}
anno_mode = mode_map[anno_choice]

# resize BEFORE drawing
disp_img, disp_det = resize_image_and_detections(
    image, detections, target_w=display_w
)

font_size = max(14, display_w // 60)

col_left, col_right = st.columns([1.2, 1.0], gap="large")

with col_left:
    st.subheader("Image")
    if anno_mode is None:
        st.image(disp_img, width=display_w)
    else:
        annotated = draw_annotated(
            disp_img,
            disp_det,
            mode=anno_mode,  # type: ignore[arg-type]
            font_size=font_size,
        )
        st.image(annotated, width=display_w)

with col_right:
    st.subheader("Results")

    a, b, c = st.columns(3)
    a.metric("Cells", summary["total"])
    b.metric("Mitotic", summary["mitotic"])
    c.metric("Non-mitotic", summary["non_mitotic"])

    st.metric("Mitotic index (%)", f"{summary['mitotic_index']:.2f}")

    if download_csv:
        csv_bytes = detections.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download detections CSV",
            data=csv_bytes,
            file_name="allium_cepa_detections.csv",
            mime="text/csv",
            use_container_width=True,
        )

if show_table:
    st.subheader("Detections table")
    st.dataframe(detections, use_container_width=True)
