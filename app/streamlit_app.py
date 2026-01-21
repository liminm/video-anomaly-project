import base64
import os
import time
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "1800"))
INLINE_GIF_MAX_MB = float(os.getenv("INLINE_GIF_MAX_MB", "5"))
INLINE_GIF_MAX_BYTES = int(INLINE_GIF_MAX_MB * 1024 * 1024)

st.set_page_config(page_title="UCSD LSTM Anomaly Explorer", layout="wide")

st.title("UCSD LSTM Anomaly Explorer")

st.sidebar.header("Server")
st.sidebar.text(f"API: {API_URL}")
st.sidebar.text(f"Timeout: {API_TIMEOUT}s")

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None
if "last_clip" not in st.session_state:
    st.session_state.last_clip = None
if "last_stride" not in st.session_state:
    st.session_state.last_stride = None
if "last_gif_stride" not in st.session_state:
    st.session_state.last_gif_stride = None
if "last_max_frames" not in st.session_state:
    st.session_state.last_max_frames = None
if "last_gif_scale" not in st.session_state:
    st.session_state.last_gif_scale = None
if "last_gif_max_frames" not in st.session_state:
    st.session_state.last_gif_max_frames = None


def format_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


@st.cache_data(ttl=30)
def fetch_clips():
    last_exc = None
    for _ in range(5):
        try:
            resp = requests.get(f"{API_URL}/clips", timeout=10)
            resp.raise_for_status()
            return resp.json()["clips"]
        except Exception as exc:
            last_exc = exc
            time.sleep(2)
    raise last_exc


try:
    clips = fetch_clips()
except Exception as exc:
    st.error(f"Failed to load clips: {exc}")
    st.stop()

clip = st.selectbox("Select a clip", clips)

quick_mode = st.toggle("Quick mode (every 4th frame)", value=False)
manual_stride = st.slider(
    "Processing stride",
    min_value=1,
    max_value=6,
    value=1,
    help="Higher stride is faster but skips frames.",
    disabled=quick_mode,
)
stride = 4 if quick_mode else manual_stride

max_frames = st.number_input(
    "Max frames (0 = all)",
    min_value=0,
    max_value=1000,
    value=0,
    step=50,
    help="Limit frames per request to reduce latency.",
)
max_frames_payload = None if int(max_frames) == 0 else int(max_frames)
save_gif = st.toggle("Generate GIF", value=True)
gif_stride = st.slider(
    "GIF frame step",
    min_value=1,
    max_value=8,
    value=1,
    help="Save every Nth processed frame to the GIF to reduce memory.",
    disabled=not save_gif,
)
gif_scale = st.slider(
    "GIF scale",
    min_value=0.25,
    max_value=1.0,
    value=1.0,
    step=0.05,
    help="Downscale the GIF to reduce memory and file size.",
    disabled=not save_gif,
)
gif_max_frames = st.number_input(
    "Max GIF frames (0 = all)",
    min_value=0,
    max_value=1000,
    value=0,
    step=50,
    help="Limit how many frames are written to the GIF.",
    disabled=not save_gif,
)
gif_max_frames_payload = None if int(gif_max_frames) == 0 else int(gif_max_frames)

if st.button("Run detection"):
    st.session_state.last_error = None
    st.session_state.last_result = None
    with st.spinner("Running inference..."):
        try:
            resp = requests.post(
                f"{API_URL}/analyze",
                json={
                    "clip": clip,
                    "save_gif": save_gif,
                    "stride": stride,
                    "gif_stride": gif_stride,
                    "max_frames": max_frames_payload,
                    "gif_scale": gif_scale,
                    "gif_max_frames": gif_max_frames_payload,
                },
                timeout=API_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            st.session_state.last_result = data
            st.session_state.last_clip = clip
            st.session_state.last_stride = stride
            st.session_state.last_gif_stride = gif_stride
            st.session_state.last_max_frames = max_frames_payload
            st.session_state.last_gif_scale = gif_scale
            st.session_state.last_gif_max_frames = gif_max_frames_payload
        except requests.exceptions.Timeout:
            st.session_state.last_error = (
                f"Request timed out after {API_TIMEOUT}s. "
                "Try quick mode or increase API_TIMEOUT."
            )
        except Exception as exc:
            st.session_state.last_error = f"Request failed: {exc}"

if st.session_state.last_error:
    st.error(st.session_state.last_error)

if st.session_state.last_result:
    data = st.session_state.last_result
    clip_label = st.session_state.last_clip or clip
    stride_label = data.get("stride", st.session_state.last_stride or stride)
    gif_stride_label = st.session_state.last_gif_stride or gif_stride
    max_frames_label = st.session_state.last_max_frames
    gif_scale_label = st.session_state.last_gif_scale or gif_scale
    gif_max_frames_label = st.session_state.last_gif_max_frames

    st.subheader("Results")
    st.write(f"Clip: {clip_label}")
    st.write(f"Max score: {data['max_score']:.2f}")
    st.write(f"Alarm frames: {len(data['alarm_frames'])}")
    st.write(f"Stride: {stride_label}")
    st.write(f"GIF step: {gif_stride_label}")
    st.write(f"Max frames: {max_frames_label or 'all'}")
    st.write(f"GIF scale: {gif_scale_label:.2f}")
    st.write(f"GIF max frames: {gif_max_frames_label or 'all'}")

    if data.get("scores"):
        st.line_chart(data["scores"], height=200)

    gif_path = data.get("gif_path")
    gif_url = data.get("gif_url")

    local_path = Path(gif_path) if gif_path else None
    if local_path and local_path.exists():
        size = local_path.stat().st_size
        if size == 0:
            st.warning("GIF file is empty.")
        elif size <= INLINE_GIF_MAX_BYTES:
            try:
                b64 = base64.b64encode(local_path.read_bytes()).decode("ascii")
                st.markdown(
                    f'<img src="data:image/gif;base64,{b64}" alt="Anomaly visualization">',
                    unsafe_allow_html=True,
                )
                st.caption(f"GIF size: {format_bytes(size)} (inlined)")
            except Exception as exc:
                st.warning(f"Failed to inline GIF: {exc}")
                st.image(str(local_path), caption="Anomaly visualization")
                st.caption(f"GIF size: {format_bytes(size)}")
        else:
            st.image(str(local_path), caption="Anomaly visualization")
            st.caption(
                f"GIF size: {format_bytes(size)} (too large to inline)"
            )
    elif gif_url:
        try:
            resp = requests.get(f"{API_URL}{gif_url}", timeout=30)
            resp.raise_for_status()
            content = resp.content
            if not content.startswith(b"GIF"):
                raise ValueError(f"Unexpected GIF header: {content[:10]!r}")
            st.image(content, caption="Anomaly visualization")
        except Exception as exc:
            if gif_path and os.path.exists(gif_path):
                st.image(gif_path, caption="Anomaly visualization")
            else:
                st.warning(f"Failed to load GIF: {exc}")
    elif gif_path and os.path.exists(gif_path):
        st.image(gif_path, caption="Anomaly visualization")
    else:
        st.info("No visualization returned.")
