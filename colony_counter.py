from __future__ import annotations

import argparse
import csv
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class CounterParams:
    max_side: int = 1800
    plate_margin: float = 0.91
    min_area: int = 5
    max_area: int = 2500
    min_circularity: float = 0.08
    yellow_gain: int = 18
    lightness_min: int = 95
    value_min: int = 105
    cream_lightness_min: int = 115
    cream_saturation_max: int = 120
    red_blue_delta: int = 8
    local_contrast: int = 0
    top_hat_kernel: int = 25
    open_iterations: int = 1
    close_iterations: int = 0
    watershed_core_ratio: float = 0.34
    watershed_core_max_distance: float = 8.0
    background_dilation: int = 1
    average_colony_area: float = 62.0
    clump_area_factor: float = 2.6
    enable_clump_estimate: bool = True


@dataclass(frozen=True)
class PlateCircle:
    cx: int
    cy: int
    radius: int
    confidence: float
    source: str = "auto"


@dataclass(frozen=True)
class Colony:
    id: int
    x: float
    y: float
    radius: float
    area: float
    circularity: float
    estimated_count: int


@dataclass
class CountResult:
    total_count: int
    detected_objects: int
    clump_extra: int
    plate: PlateCircle
    resize_scale: float
    original_size: tuple[int, int]
    work_size: tuple[int, int]
    colonies: list[Colony]
    overlay_bgr: np.ndarray
    colony_mask: np.ndarray
    plate_mask: np.ndarray


def load_image_bgr(source: str | Path | bytes | BinaryIO) -> np.ndarray:
    if isinstance(source, (str, Path)):
        image = Image.open(source)
    elif isinstance(source, bytes):
        image = Image.open(io.BytesIO(source))
    else:
        image = Image.open(source)

    image = ImageOps.exif_transpose(image).convert("RGB")
    image_rgb = np.array(image)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def resize_for_work(img_bgr: np.ndarray, max_side: int = 1800) -> tuple[np.ndarray, float]:
    height, width = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return img_bgr.copy(), 1.0

    size = (int(width * scale), int(height * scale))
    resized = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    return resized, scale


def find_plate_circle(img_bgr: np.ndarray) -> PlateCircle:
    height, width = img_bgr.shape[:2]
    min_side = min(width, height)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 1.5)

    min_radius = int(min_side * 0.15)
    max_radius = int(min_side * 0.48)
    candidates: list[tuple[int, int, int]] = []

    for threshold in (34, 28, 22):
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(1, min_side // 3),
            param1=80,
            param2=threshold,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is not None:
            candidates.extend(tuple(map(int, circle)) for circle in np.round(circles[0]).astype(int))
            break

    if not candidates:
        return PlateCircle(
            cx=int(width * 0.52),
            cy=int(height * 0.53),
            radius=int(min_side * 0.25),
            confidence=0.2,
            source="fallback",
        )

    center_x = width / 2
    center_y = height / 2

    def score(circle: tuple[int, int, int]) -> float:
        x, y, radius = circle
        center_penalty = math.hypot((x - center_x) / width, (y - center_y) / height)
        outside = (
            max(0, radius - x)
            + max(0, radius - y)
            + max(0, x + radius - width)
            + max(0, y + radius - height)
        ) / max(radius, 1)
        radius_bonus = 0.12 * (radius / min_side)
        return center_penalty + outside - radius_bonus

    best = min(candidates, key=score)
    best_score = score(best)
    confidence = max(0.35, min(0.99, 0.98 - best_score))
    return PlateCircle(best[0], best[1], best[2], round(confidence, 3), "hough")


def make_plate_mask(shape: tuple[int, int], plate: PlateCircle, margin: float) -> np.ndarray:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    radius = max(1, int(plate.radius * margin))
    mask = ((xx - plate.cx) ** 2 + (yy - plate.cy) ** 2 <= radius**2).astype(np.uint8)
    return mask * 255


def build_colony_mask(
    img_bgr: np.ndarray,
    plate: PlateCircle,
    params: CounterParams,
) -> tuple[np.ndarray, np.ndarray]:
    plate_mask = make_plate_mask(img_bgr.shape[:2], plate, params.plate_margin)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blue, _, red = cv2.split(img_bgr)
    lightness, a_channel, b_channel = cv2.split(lab)
    _, saturation, value = cv2.split(hsv)

    yellowish = (
        (b_channel.astype(np.int16) - a_channel.astype(np.int16) > params.yellow_gain)
        & (lightness > params.lightness_min)
        & (value > params.value_min)
    )
    creamy = (
        (lightness > params.cream_lightness_min)
        & (saturation < params.cream_saturation_max)
        & (red.astype(np.int16) > blue.astype(np.int16) + params.red_blue_delta)
    )
    not_glare = ~((value > 235) & (saturation < 35))
    colony_pixels = (yellowish | creamy) & not_glare

    if params.local_contrast > 0:
        kernel_size = max(5, int(params.top_hat_kernel))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        local_bright = (
            (top_hat > params.local_contrast)
            & (lightness > max(40, params.lightness_min - 15))
            & (value > max(40, params.value_min - 20))
            & (saturation < 190)
            & not_glare
        )
        colony_pixels = colony_pixels | local_bright

    mask = colony_pixels.astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, plate_mask)

    if params.open_iterations > 0:
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=params.open_iterations)

    if params.close_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=params.close_iterations)

    return mask, plate_mask


def watershed_split(img_bgr: np.ndarray, mask: np.ndarray, params: CounterParams) -> np.ndarray:
    if cv2.countNonZero(mask) == 0:
        return np.zeros(mask.shape, dtype=np.int32)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(mask, kernel, iterations=max(1, params.background_dilation))
    ratio = min(0.95, max(0.01, params.watershed_core_ratio))

    component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    sure_fg = np.zeros(mask.shape, dtype=np.uint8)

    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < 3:
            continue

        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        component = (component_labels[y : y + height, x : x + width] == component_id).astype(np.uint8) * 255
        distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        max_distance = float(distance.max())
        if max_distance <= 0:
            continue

        threshold = max(1.0, ratio * max_distance)
        if params.watershed_core_max_distance > 0:
            threshold = min(threshold, params.watershed_core_max_distance)

        local_fg = (distance >= threshold).astype(np.uint8) * 255
        if cv2.countNonZero(local_fg) == 0:
            _, _, _, max_location = cv2.minMaxLoc(distance)
            local_fg[max_location[1], max_location[0]] = 255

        sure_fg[y : y + height, x : x + width] = cv2.bitwise_or(
            sure_fg[y : y + height, x : x + width],
            local_fg,
        )

    if cv2.countNonZero(sure_fg) == 0:
        sure_fg = mask.copy()

    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img_bgr, markers)
    return markers


def count_colonies(
    img_bgr: np.ndarray,
    params: CounterParams | None = None,
    plate: PlateCircle | None = None,
    resize_scale: float = 1.0,
    original_size: tuple[int, int] | None = None,
    show_labels: bool = False,
) -> CountResult:
    params = params or CounterParams()
    plate = plate or find_plate_circle(img_bgr)
    original_size = original_size or (img_bgr.shape[1], img_bgr.shape[0])

    colony_mask, plate_mask = build_colony_mask(img_bgr, plate, params)
    markers = watershed_split(img_bgr, colony_mask, params)

    colonies: list[Colony] = []
    total_count = 0
    next_id = 1
    unique_markers = np.unique(markers)

    for marker in unique_markers:
        if marker <= 1:
            continue

        component = np.zeros(colony_mask.shape, dtype=np.uint8)
        component[markers == marker] = 255
        pixel_area = float(cv2.countNonZero(component))
        if pixel_area < params.min_area or pixel_area > params.max_area:
            continue

        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        contour_area = max(float(cv2.contourArea(contour)), 1.0)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4.0 * math.pi * contour_area / (perimeter * perimeter) if perimeter else 0.0
        if circularity < params.min_circularity:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            ys, xs = np.nonzero(component)
            if len(xs) == 0:
                continue
            x = float(xs.mean())
            y = float(ys.mean())
        else:
            x = float(moments["m10"] / moments["m00"])
            y = float(moments["m01"] / moments["m00"])

        if not (0 <= int(round(y)) < plate_mask.shape[0] and 0 <= int(round(x)) < plate_mask.shape[1]):
            continue
        if plate_mask[int(round(y)), int(round(x))] == 0:
            continue

        radius = math.sqrt(pixel_area / math.pi)
        estimated_count = 1
        if params.enable_clump_estimate and pixel_area > params.average_colony_area * params.clump_area_factor:
            estimated_count = max(1, int(round(pixel_area / params.average_colony_area)))

        colonies.append(
            Colony(
                id=next_id,
                x=round(x, 2),
                y=round(y, 2),
                radius=round(radius, 2),
                area=round(pixel_area, 2),
                circularity=round(circularity, 3),
                estimated_count=estimated_count,
            )
        )
        total_count += estimated_count
        next_id += 1

    overlay = draw_overlay(img_bgr, plate, params, colonies, total_count, show_labels=show_labels)
    return CountResult(
        total_count=total_count,
        detected_objects=len(colonies),
        clump_extra=total_count - len(colonies),
        plate=plate,
        resize_scale=resize_scale,
        original_size=original_size,
        work_size=(img_bgr.shape[1], img_bgr.shape[0]),
        colonies=colonies,
        overlay_bgr=overlay,
        colony_mask=colony_mask,
        plate_mask=plate_mask,
    )


def draw_overlay(
    img_bgr: np.ndarray,
    plate: PlateCircle,
    params: CounterParams,
    colonies: Iterable[Colony],
    total_count: int,
    show_labels: bool = False,
) -> np.ndarray:
    overlay = img_bgr.copy()
    cv2.circle(overlay, (plate.cx, plate.cy), int(plate.radius * params.plate_margin), (0, 210, 255), 2)
    cv2.circle(overlay, (plate.cx, plate.cy), 3, (0, 210, 255), -1)

    for colony in colonies:
        center = (int(round(colony.x)), int(round(colony.y)))
        radius = max(3, int(round(colony.radius)))
        color = (60, 220, 60) if colony.estimated_count == 1 else (0, 140, 255)
        cv2.circle(overlay, center, radius, color, 1)
        if colony.estimated_count > 1:
            cv2.putText(
                overlay,
                str(colony.estimated_count),
                (center[0] + 4, center[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                color,
                1,
                cv2.LINE_AA,
            )
        elif show_labels:
            cv2.putText(
                overlay,
                str(colony.id),
                (center[0] + 3, center[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                color,
                1,
                cv2.LINE_AA,
            )

    label = f"CFU = {total_count}"
    cv2.rectangle(overlay, (18, 18), (230, 64), (255, 255, 255), -1)
    cv2.rectangle(overlay, (18, 18), (230, 64), (0, 0, 0), 1)
    cv2.putText(overlay, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return overlay


def analyze_image(
    img_bgr: np.ndarray,
    params: CounterParams | None = None,
    plate: PlateCircle | None = None,
    show_labels: bool = False,
) -> CountResult:
    params = params or CounterParams()
    work_bgr, scale = resize_for_work(img_bgr, params.max_side)
    original_size = (img_bgr.shape[1], img_bgr.shape[0])
    if plate is None:
        plate = find_plate_circle(work_bgr)
    return count_colonies(
        work_bgr,
        params=params,
        plate=plate,
        resize_scale=scale,
        original_size=original_size,
        show_labels=show_labels,
    )


def analyze_file(
    image_path: str | Path,
    params: CounterParams | None = None,
    show_labels: bool = False,
) -> CountResult:
    image = load_image_bgr(image_path)
    return analyze_image(image, params=params, show_labels=show_labels)


def colonies_to_records(result: CountResult) -> list[dict[str, float | int]]:
    scale = result.resize_scale or 1.0
    records = []
    for colony in result.colonies:
        record = asdict(colony)
        record["x_original"] = round(colony.x / scale, 2)
        record["y_original"] = round(colony.y / scale, 2)
        record["radius_original"] = round(colony.radius / scale, 2)
        records.append(record)
    return records


def result_summary(result: CountResult) -> dict[str, object]:
    return {
        "total_count": result.total_count,
        "detected_objects": result.detected_objects,
        "clump_extra": result.clump_extra,
        "resize_scale": result.resize_scale,
        "original_size": result.original_size,
        "work_size": result.work_size,
        "plate": asdict(result.plate),
    }


def encode_png(img_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Failed to encode PNG")
    return encoded.tobytes()


def write_csv(path: str | Path, records: list[dict[str, float | int]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "x",
        "y",
        "radius",
        "area",
        "circularity",
        "estimated_count",
        "x_original",
        "y_original",
        "radius_original",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_png(path: str | Path, img_bgr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img_bgr):
        raise RuntimeError(f"Failed to write image: {path}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Count colonies in a petri dish image.")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--overlay", help="Write annotated PNG")
    parser.add_argument("--csv", help="Write colony detail CSV")
    parser.add_argument("--json", help="Write summary JSON")
    parser.add_argument("--max-side", type=int, default=CounterParams.max_side)
    parser.add_argument("--plate-margin", type=float, default=CounterParams.plate_margin)
    parser.add_argument("--min-area", type=int, default=CounterParams.min_area)
    parser.add_argument("--max-area", type=int, default=CounterParams.max_area)
    parser.add_argument("--yellow-gain", type=int, default=CounterParams.yellow_gain)
    parser.add_argument("--watershed-core-ratio", type=float, default=CounterParams.watershed_core_ratio)
    parser.add_argument("--watershed-core-max-distance", type=float, default=CounterParams.watershed_core_max_distance)
    parser.add_argument("--local-contrast", type=int, default=CounterParams.local_contrast)
    parser.add_argument("--average-colony-area", type=float, default=CounterParams.average_colony_area)
    parser.add_argument("--clump-estimate", action="store_true")
    parser.add_argument("--labels", action="store_true", help="Draw colony IDs on overlay")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    params = CounterParams(
        max_side=args.max_side,
        plate_margin=args.plate_margin,
        min_area=args.min_area,
        max_area=args.max_area,
        yellow_gain=args.yellow_gain,
        watershed_core_ratio=args.watershed_core_ratio,
        watershed_core_max_distance=args.watershed_core_max_distance,
        local_contrast=args.local_contrast,
        average_colony_area=args.average_colony_area,
        enable_clump_estimate=args.clump_estimate,
    )
    result = analyze_file(args.image, params=params, show_labels=args.labels)
    records = colonies_to_records(result)

    if args.overlay:
        write_png(args.overlay, result.overlay_bgr)
    if args.csv:
        write_csv(args.csv, records)
    if args.json:
        output = result_summary(result)
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result_summary(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
