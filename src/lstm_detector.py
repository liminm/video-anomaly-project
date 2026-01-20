from collections import deque
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

FRAME_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
CACHE_DIR_DEFAULT = Path("/tmp/ucsd_cache")

# Detection config (mirrors detect_lstm.py)
BOX_SIZE = 40
THRESH_TRIGGER_COUNT = 200
THRESH_RESET_COUNT = 100
THRESH_VIZ_BOX = 180


def _list_frames(clip_dir: Path):
    for ext in FRAME_EXTS:
        files = sorted(clip_dir.glob(f"*{ext}"))
        if files:
            return files
    return []


def _list_local_clips(root_dir: Path):
    clips = []
    for path in sorted(root_dir.iterdir()):
        if not path.is_dir():
            continue
        if _list_frames(path):
            clips.append(path.name)
    return clips


def _normalize_prefix(prefix: str | None):
    if not prefix:
        return ""
    return prefix.strip("/") + "/"


def _get_gcs_client():
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise ImportError(
            "google-cloud-storage is required when using GCS data"
        ) from exc
    return storage.Client()


def _list_gcs_clips(bucket_name: str, prefix: str | None):
    client = _get_gcs_client()
    prefix = _normalize_prefix(prefix)
    iterator = client.list_blobs(bucket_name, prefix=prefix, delimiter="/")

    prefixes = set()
    for page in iterator.pages:
        prefixes.update(page.prefixes)

    clips = []
    for p in sorted(prefixes):
        name = p.rstrip("/").split("/")[-1]
        if name:
            clips.append(name)

    return clips


def list_available_clips(
    root_dir: Path,
    gcs_bucket: str | None = None,
    gcs_prefix: str | None = None,
):
    if gcs_bucket:
        return _list_gcs_clips(gcs_bucket, gcs_prefix)
    return _list_local_clips(root_dir)


def _download_gcs_clip(
    bucket_name: str,
    prefix: str | None,
    clip: str,
    dest_dir: Path,
):
    client = _get_gcs_client()
    prefix = _normalize_prefix(prefix)
    clip_prefix = f"{prefix}{clip}/"

    dest_dir.mkdir(parents=True, exist_ok=True)

    found = False
    for blob in client.list_blobs(bucket_name, prefix=clip_prefix):
        rel = blob.name[len(clip_prefix) :]
        if not rel or rel.endswith("/"):
            continue
        found = True
        local_path = dest_dir / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)

    if not found:
        raise ValueError(f"Clip not found in GCS: {clip}")


def resolve_clip_dir(
    clip: str,
    root_dir: Path,
    gcs_bucket: str | None = None,
    gcs_prefix: str | None = None,
    cache_dir: Path | None = None,
):
    if not gcs_bucket:
        return root_dir / clip

    cache_root = cache_dir or CACHE_DIR_DEFAULT
    local_dir = cache_root / clip
    if _list_frames(local_dir):
        return local_dir

    _download_gcs_clip(gcs_bucket, gcs_prefix, clip, local_dir)

    if not _list_frames(local_dir):
        raise ValueError(f"No frames found after download for clip: {clip}")

    return local_dir


def _load_gray_frame(path: Path, size: tuple[int, int]):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read frame: {path}")
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0


class LSTMAnomalyDetector:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        seq_len = input_shape[1] if isinstance(input_shape[1], int) else None
        height = input_shape[3] if isinstance(input_shape[3], int) else None
        width = input_shape[4] if isinstance(input_shape[4], int) else None

        if seq_len is None or height is None or width is None:
            raise ValueError("ONNX model input shape must be static for seq/h/w")

        self.seq_len = seq_len
        self.size = (width, height)

    def analyze_clip(
        self,
        clip_dir: Path,
        output_dir: Path,
        save_gif: bool = True,
        stride: int = 1,
        gif_stride: int = 1,
    ):
        files = _list_frames(clip_dir)
        if len(files) < 2:
            raise ValueError(f"Not enough frames in {clip_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        gif_writer = None
        gif_path = None
        gif_frame_index = 0
        gif_frame_count = 0
        scores = []
        alarm_frames = []
        alarm_active = False

        kernel_close = np.ones((5, 5), np.uint8)
        smooth_density_map = np.zeros((self.size[1], self.size[0]), dtype=np.float32)

        window = deque(maxlen=self.seq_len)
        prev_raw = None

        if stride < 1:
            raise ValueError("stride must be >= 1")
        if gif_stride < 1:
            raise ValueError("gif_stride must be >= 1")

        for i, path in enumerate(files):
            if i % stride != 0:
                continue
            raw = _load_gray_frame(path, self.size)

            if prev_raw is not None and len(window) == self.seq_len:
                input_seq = np.stack(window, axis=0)
                input_seq = np.expand_dims(input_seq, axis=0)

                pred_seq = self.session.run(
                    [self.output_name], {self.input_name: input_seq}
                )[0]
                pred = pred_seq[0, -1, 0, :, :]

                diff = np.abs(raw - pred)
                mask = (np.abs(raw - prev_raw) > 0.04).astype(np.float32)
                masked_diff = diff * mask

                _, binary_map = cv2.threshold(
                    masked_diff, 0.10, 1.0, cv2.THRESH_BINARY
                )
                binary_map_uint8 = (binary_map * 255).astype(np.uint8)
                closed_map = cv2.morphologyEx(
                    binary_map_uint8, cv2.MORPH_CLOSE, kernel_close
                )

                density_map_raw = cv2.boxFilter(
                    closed_map / 255.0,
                    -1,
                    (BOX_SIZE, BOX_SIZE),
                    normalize=False,
                )
                smooth_density_map = cv2.addWeighted(
                    smooth_density_map, 0.6, density_map_raw.astype(np.float32), 0.4, 0
                )
                score = float(np.max(smooth_density_map))
                scores.append(score)

                if not alarm_active:
                    if score > THRESH_TRIGGER_COUNT:
                        alarm_active = True
                else:
                    if score < THRESH_RESET_COUNT:
                        alarm_active = False

                if alarm_active:
                    alarm_frames.append(i)

                if save_gif:
                    write_frame = gif_frame_index % gif_stride == 0
                    if write_frame:
                        img_cv = cv2.imread(str(path))
                        img_cv = cv2.resize(img_cv, self.size)

                        heatmap_norm = np.clip(
                            smooth_density_map / float(THRESH_TRIGGER_COUNT) * 255, 0, 255
                        ).astype(np.uint8)
                        heatmap = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

                        vis_hud = img_cv.copy()
                        overlay = vis_hud.copy()

                        mask_paint = (density_map_raw > 100).astype(np.uint8)
                        precise_mask = cv2.bitwise_and(
                            closed_map, closed_map, mask=mask_paint
                        )
                        contours_paint, _ = cv2.findContours(
                            precise_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )

                        for cnt in contours_paint:
                            if cv2.contourArea(cnt) > 20:
                                hull = cv2.convexHull(cnt)
                                cv2.drawContours(overlay, [hull], -1, (0, 0, 255), -1)

                        cv2.addWeighted(overlay, 0.4, vis_hud, 0.6, 0, vis_hud)

                        mask_box = (smooth_density_map > THRESH_VIZ_BOX).astype(np.uint8)
                        contours_box, _ = cv2.findContours(
                            mask_box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )

                        for cnt in contours_box:
                            if cv2.contourArea(cnt) > 50:
                                x, y, w, h = cv2.boundingRect(cnt)
                                color = (0, 0, 255)

                                d = 10
                                t = 2
                                cv2.line(vis_hud, (x, y), (x + d, y), color, t)
                                cv2.line(vis_hud, (x, y), (x, y + d), color, t)
                                cv2.line(vis_hud, (x + w, y), (x + w - d, y), color, t)
                                cv2.line(vis_hud, (x + w, y), (x + w, y + d), color, t)
                                cv2.line(vis_hud, (x, y + h), (x + d, y + h), color, t)
                                cv2.line(vis_hud, (x, y + h), (x, y + h - d), color, t)
                                cv2.line(
                                    vis_hud, (x + w, y + h), (x + w - d, y + h), color, t
                                )
                                cv2.line(
                                    vis_hud, (x + w, y + h), (x + w, y + h - d), color, t
                                )

                                cv2.putText(
                                    vis_hud,
                                    "ANOMALY",
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    1,
                                )

                        status = "ALARM ACTIVE" if alarm_active else "MONITORING"
                        border_color = (0, 0, 255) if alarm_active else (0, 255, 0)

                        if alarm_active:
                            for panel in (heatmap, vis_hud):
                                cv2.rectangle(panel, (0, 0), (255, 255), border_color, 5)

                        combined = np.hstack((heatmap, vis_hud))
                        pil_img = Image.fromarray(
                            cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
                        )
                        draw = ImageDraw.Draw(pil_img)
                        draw.text((10, 10), f"Density Heatmap ({score:.0f})", fill="white")
                        draw.text((266, 10), "Inference Output", fill="white")
                        draw.text((266, 230), status, fill=border_color[::-1])

                        if gif_writer is None:
                            output_name = f"lstm_{clip_dir.name}.gif"
                            gif_path = output_dir / output_name
                            gif_writer = imageio.get_writer(
                                gif_path, mode="I", duration=0.1
                            )
                        gif_writer.append_data(np.asarray(pil_img))
                        gif_frame_count += 1

                    gif_frame_index += 1

            window.append(raw[np.newaxis, ...])
            prev_raw = raw

        if gif_writer is not None:
            gif_writer.close()
            if gif_frame_count == 0 and gif_path is not None:
                try:
                    gif_path.unlink()
                except OSError:
                    pass
                gif_path = None

        return {
            "clip": clip_dir.name,
            "scores": scores,
            "max_score": max(scores) if scores else 0.0,
            "alarm_frames": alarm_frames,
            "gif_path": str(gif_path) if gif_path else None,
            "stride": stride,
        }
