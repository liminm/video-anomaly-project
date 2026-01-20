import os
import time

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "1800"))

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
stride = 4 if quick_mode else 1
save_gif = st.toggle("Generate GIF", value=True)
gif_stride = st.slider(
    "GIF frame step",
    min_value=1,
    max_value=8,
    value=1,
    help="Save every Nth processed frame to the GIF to reduce memory.",
    disabled=not save_gif,
)

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
                },
                timeout=API_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            st.session_state.last_result = data
            st.session_state.last_clip = clip
            st.session_state.last_stride = stride
            st.session_state.last_gif_stride = gif_stride
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

    st.subheader("Results")
    st.write(f"Clip: {clip_label}")
    st.write(f"Max score: {data['max_score']:.2f}")
    st.write(f"Alarm frames: {len(data['alarm_frames'])}")
    st.write(f"Stride: {stride_label}")
    st.write(f"GIF step: {gif_stride_label}")

    if data.get("scores"):
        st.line_chart(data["scores"], height=200)

    gif_path = data.get("gif_path")
    gif_url = data.get("gif_url")

    if gif_url:
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
