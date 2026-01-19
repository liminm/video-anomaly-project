import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="UCSD LSTM Anomaly Explorer", layout="wide")

st.title("UCSD LSTM Anomaly Explorer")

st.sidebar.header("Server")
st.sidebar.text(f"API: {API_URL}")


@st.cache_data(ttl=30)
def fetch_clips():
    resp = requests.get(f"{API_URL}/clips", timeout=10)
    resp.raise_for_status()
    return resp.json()["clips"]


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

    if gif_path and os.path.exists(gif_path):
        st.image(gif_path, caption="Anomaly visualization")
    elif gif_url:
        st.image(f"{API_URL}{gif_url}")
    else:
        st.info("No visualization returned.")
