from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from colony_counter import (
    CounterParams,
    PlateCircle,
    analyze_image,
    colonies_to_records,
    encode_png,
    find_plate_circle,
    load_image_bgr,
    resize_for_work,
    result_summary,
)


ROOT = Path(__file__).resolve().parent
SAMPLE_IMAGE = ROOT / "samples" / "sample_plate.jpg"


st.set_page_config(page_title="菌落计数工具", layout="wide")

st.title("菌落计数工具")
st.caption("自动定位培养皿，识别乳白/淡黄菌落，输出 CFU、标注图和 CSV。")


@st.cache_data(show_spinner=False)
def prepare_image(image_bytes: bytes, max_side: int) -> tuple[np.ndarray, np.ndarray, float, PlateCircle]:
    original = load_image_bgr(image_bytes)
    work, scale = resize_for_work(original, max_side=max_side)
    plate = find_plate_circle(work)
    return original, work, scale, plate


def image_source() -> tuple[bytes | None, str]:
    uploaded = st.sidebar.file_uploader("上传培养皿图片", type=["jpg", "jpeg", "png", "webp", "tif", "tiff"])
    if uploaded is not None:
        return uploaded.getvalue(), uploaded.name

    if SAMPLE_IMAGE.exists():
        return SAMPLE_IMAGE.read_bytes(), SAMPLE_IMAGE.name

    return None, ""


def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


with st.sidebar:
    st.header("图像")

image_bytes, source_name = image_source()

if image_bytes is None:
    st.info("请先在左侧上传一张培养皿图片。")
    st.stop()

base_params = CounterParams()
original_bgr, work_bgr, scale, auto_plate = prepare_image(image_bytes, base_params.max_side)
work_height, work_width = work_bgr.shape[:2]

with st.sidebar:
    st.caption(f"当前图像：{source_name}")
    st.header("培养皿")
    plate_margin = st.slider("计数区域比例", 0.70, 0.98, base_params.plate_margin, 0.01)
    manual_plate = st.toggle("手动微调圆形区域", value=False)

    plate = auto_plate
    if manual_plate:
        center_x = st.slider("圆心 X", 0, work_width - 1, int(auto_plate.cx), 1)
        center_y = st.slider("圆心 Y", 0, work_height - 1, int(auto_plate.cy), 1)
        min_radius = max(20, int(min(work_width, work_height) * 0.08))
        max_radius = max(min_radius + 1, int(min(work_width, work_height) * 0.55))
        radius = st.slider("半径", min_radius, max_radius, int(auto_plate.radius), 1)
        plate = PlateCircle(center_x, center_y, radius, confidence=1.0, source="manual")

    st.header("识别")
    min_area = st.slider("最小面积", 2, 120, base_params.min_area, 1)
    max_area = st.slider("最大面积", 80, 8000, base_params.max_area, 20)
    yellow_gain = st.slider("颜色阈值", 0, 80, base_params.yellow_gain, 1)
    local_contrast = st.slider("局部亮点增强", 0, 45, base_params.local_contrast, 1)
    watershed_core_ratio = st.slider("分水岭核心阈值", 0.18, 0.60, base_params.watershed_core_ratio, 0.01)
    min_circularity = st.slider("圆度过滤", 0.00, 0.45, base_params.min_circularity, 0.01)
    close_iterations = st.slider("近邻连接", 0, 2, base_params.close_iterations, 1)

    st.header("输出")
    enable_clump_estimate = st.checkbox("大面积粘连补偿", value=base_params.enable_clump_estimate)
    average_colony_area = st.slider("平均单菌落面积", 15, 220, int(base_params.average_colony_area), 1)
    correction = st.number_input("手动修正", min_value=-10000, max_value=10000, value=0, step=1)
    show_labels = st.checkbox("显示编号", value=False)
    show_debug = st.checkbox("显示中间图", value=False)

params = replace(
    base_params,
    plate_margin=plate_margin,
    min_area=min_area,
    max_area=max_area,
    yellow_gain=yellow_gain,
    local_contrast=local_contrast,
    watershed_core_ratio=watershed_core_ratio,
    min_circularity=min_circularity,
    close_iterations=close_iterations,
    enable_clump_estimate=enable_clump_estimate,
    average_colony_area=float(average_colony_area),
)

with st.spinner("正在计数..."):
    result = analyze_image(original_bgr, params=params, plate=plate, show_labels=show_labels)

final_total = max(0, result.total_count + int(correction))
records = colonies_to_records(result)
df = pd.DataFrame(records)

metric_cols = st.columns(5)
metric_cols[0].metric("最终计数", f"{final_total} CFU")
metric_cols[1].metric("自动计数", f"{result.total_count} CFU")
metric_cols[2].metric("分割对象", result.detected_objects)
metric_cols[3].metric("粘连补偿", result.clump_extra)
metric_cols[4].metric("培养皿置信度", f"{result.plate.confidence:.2f}")

left, right = st.columns([1.12, 0.88], gap="large")

with left:
    tabs = ["标注图", "原图"]
    if show_debug:
        tabs.extend(["菌落 Mask", "培养皿 Mask"])
    image_tabs = st.tabs(tabs)
    with image_tabs[0]:
        st.image(to_rgb(result.overlay_bgr), width="stretch")
    with image_tabs[1]:
        st.image(to_rgb(work_bgr), width="stretch")
    if show_debug:
        with image_tabs[2]:
            st.image(result.colony_mask, width="stretch", clamp=True)
        with image_tabs[3]:
            st.image(result.plate_mask, width="stretch", clamp=True)

with right:
    st.subheader("结果导出")
    summary = result_summary(result)
    summary["manual_correction"] = int(correction)
    summary["final_count"] = final_total

    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    overlay_bytes = encode_png(result.overlay_bgr)
    summary_bytes = pd.Series(summary, dtype="object").to_json(force_ascii=False, indent=2).encode("utf-8")

    d1, d2 = st.columns(2)
    d1.download_button("下载 CSV", csv_bytes, file_name="colony_count.csv", mime="text/csv", width="stretch")
    d2.download_button(
        "下载标注图",
        overlay_bytes,
        file_name="colony_overlay.png",
        mime="image/png",
        width="stretch",
    )
    st.download_button(
        "下载汇总 JSON",
        summary_bytes,
        file_name="colony_summary.json",
        mime="application/json",
        width="stretch",
    )

    st.subheader("检测明细")
    if df.empty:
        st.warning("没有检测到菌落。可以降低最小面积、降低颜色阈值，或手动微调培养皿区域。")
    else:
        st.dataframe(df, width="stretch", height=520)

st.divider()
st.markdown(
    """
#### 本地运行
```bash
pip install -r requirements.txt
streamlit run app.py
```

#### 命令行批量处理
```bash
python colony_counter.py samples/sample_plate.jpg --overlay outputs/overlay.png --csv outputs/colonies.csv --json outputs/summary.json
```
"""
)
