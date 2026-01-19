import base64
import os
import time

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")

st.set_page_config(page_title="UCSD LSTM Anomaly Explorer", layout="wide")

st.title("UCSD LSTM Anomaly Explorer")

st.sidebar.header("Server")
st.sidebar.text(f"API: {API_URL}")


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

if st.button("Run detection"):
    with st.spinner("Running inference..."):
        resp = requests.post(
            f"{API_URL}/analyze",
            json={"clip": clip, "save_gif": True, "stride": stride},
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()

    st.subheader("Results")
    st.write(f"Max score: {data['max_score']:.2f}")
    st.write(f"Alarm frames: {len(data['alarm_frames'])}")
    st.write(f"Stride: {data.get('stride', stride)}")

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
            b64 = base64.b64encode(content).decode("ascii")
            st.markdown(
                f'<img src="data:image/gif;base64,{b64}" alt="Anomaly visualization">',
                unsafe_allow_html=True,
            )
        except Exception as exc:
            if gif_path and os.path.exists(gif_path):
                st.image(gif_path, caption="Anomaly visualization")
            else:
                st.warning(f"Failed to load GIF: {exc}")
    elif gif_path and os.path.exists(gif_path):
        st.image(gif_path, caption="Anomaly visualization")
    else:
        st.info("No visualization returned.")
