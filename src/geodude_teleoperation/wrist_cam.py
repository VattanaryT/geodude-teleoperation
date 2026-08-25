"""Load bag wrist-cam JPEGs and align them onto NPZ timestamps."""
from __future__ import annotations

from pathlib import Path

import numpy as np

WRIST_CAM_TOPIC = "/wrist_cam_0/color/image_rect_raw/compressed"


def nearest_indices(query: np.ndarray, source: np.ndarray) -> np.ndarray:
    """For each query stamp, index of the nearest value in ``source``."""
    q = np.asarray(query, dtype=float).reshape(-1)
    s = np.asarray(source, dtype=float).reshape(-1)
    if s.size == 0:
        raise ValueError("source stamps empty")
    i = np.clip(np.searchsorted(s, q), 0, len(s) - 1)
    prev = np.clip(i - 1, 0, len(s) - 1)
    pick_prev = np.abs(q - s[prev]) <= np.abs(q - s[i])
    return np.where(pick_prev, prev, i)


def overlay_caption(mode: str, t_s: float, f_n: float | None) -> str:
    fn = "n/a" if f_n is None else f"{float(f_n):.1f}N"
    return f"{mode}  t={float(t_s):.2f}s  fn={fn}"


def annotate_wrist_frame(
    rgb: np.ndarray, *, mode: str, t_s: float, f_n: float | None
) -> np.ndarray:
    """Copy RGB frame and burn in mode / time / f_n. Does not modify ``rgb``."""
    import cv2

    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    scale = max(1.2, w / 400.0)
    bar_h = int(round(40 * scale))
    font = 0.65 * scale
    thick = max(2, int(round(2 * scale)))
    bar = (40, 60, 220) if mode == "contact" else (60, 180, 80)
    cv2.rectangle(bgr, (0, 0), (w, bar_h), bar, -1)
    cv2.putText(
        bgr,
        overlay_caption(mode, t_s, f_n),
        (int(10 * scale), int(bar_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font,
        (255, 255, 255),
        thick,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_jpeg(payload: bytes | memoryview | np.ndarray) -> np.ndarray:
    import cv2

    arr = np.frombuffer(payload, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode JPEG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_compressed_images(
    bag: Path | str, topic: str = WRIST_CAM_TOPIC
) -> tuple[np.ndarray, list[bytes]]:
    """Return (stamps_s, jpeg_payloads) for ``topic``. Stamps are seconds."""
    bag = Path(bag)
    if not bag.exists():
        raise FileNotFoundError(bag)
    from rclpy.serialization import deserialize_message
    from rosbag2_py import (
        ConverterOptions, SequentialReader, StorageFilter, StorageOptions,
    )
    from rosidl_runtime_py.utilities import get_message

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag), storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr",
                         output_serialization_format="cdr"),
    )
    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in topics:
        raise KeyError(f"{topic} not in {bag}: {sorted(topics)}")
    reader.set_filter(StorageFilter(topics=[topic]))
    msg_type = get_message(topics[topic])
    stamps: list[float] = []
    payloads: list[bytes] = []
    while reader.has_next():
        name, data, t_ns = reader.read_next()
        if name != topic:
            continue
        msg = deserialize_message(data, msg_type)
        stamps.append(float(t_ns) * 1e-9)
        payloads.append(bytes(msg.data))
    if not payloads:
        raise ValueError(f"no messages on {topic} in {bag}")
    return np.asarray(stamps, dtype=float), payloads


def write_wrist_cam_npz(
    bag: Path | str, dst: Path | str, topic: str = WRIST_CAM_TOPIC
) -> Path:
    """Copy JPEG payloads out of a bag. Does not modify the bag."""
    dst = Path(dst)
    stamps, payloads = load_compressed_images(bag, topic)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst,
        stamps=stamps,
        jpeg=np.array(payloads, dtype=object),
        topic=np.array(topic),
        bag=np.array(str(bag)),
    )
    return dst


def load_wrist_cam(
    path: Path | str, topic: str = WRIST_CAM_TOPIC
) -> tuple[np.ndarray, list[bytes]]:
    """Load from a wrist-cam NPZ cache or a rosbag2 directory."""
    path = Path(path)
    if path.is_file() and path.suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        stamps = np.asarray(z["stamps"], dtype=float)
        jpeg = z["jpeg"]
        payloads = [bytes(x) for x in jpeg.tolist()]
        return stamps, payloads
    return load_compressed_images(path, topic)
