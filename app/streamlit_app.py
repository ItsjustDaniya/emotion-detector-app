# Modified section of streamlit_app.py for emoji display and animation

import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from datetime import datetime
import matplotlib.pyplot as plt

from src.face_detect import get_faces
from src.emotion_model import predict_emotion, get_prediction_probs

# ---- CONFIG ----
st.set_page_config(page_title="Emotion App", layout="wide")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emoji_path = "emojis"
emojis = {
    e: Image.open(os.path.join(emoji_path, f"{e.lower()}.png")).convert("RGBA")
    for e in emotion_labels if os.path.exists(os.path.join(emoji_path, f"{e.lower()}.png"))
}

# ---- SESSION STATE INIT ----
if "emotion_counts" not in st.session_state:
    st.session_state.emotion_counts = {e: 0 for e in emotion_labels}
    st.session_state.emotion_log = []
    st.session_state.fps_log = []
    st.session_state.camera_on = False
    st.session_state.has_run_once = False

# Emoji tracking state
if "current_emoji" not in st.session_state:
    st.session_state.current_emoji = None
    st.session_state.last_emoji_change_time = time.time()

# ---- UI CONTROLS ----
st.title("Real-Time Emotion Detector")
col1, col2, col3 = st.columns([0.5, 0.3, 0.2])

with col1:
    run = st.toggle("Turn Camera On")
    st.session_state.camera_on = run
with col2:
    snapshot = st.button("Capture Snapshot")
with col3:
    if st.button("Reset Stats"):
        st.session_state.emotion_counts = {e: 0 for e in emotion_labels}
        st.session_state.fps_log = []

# ---- TABS ----
if st.session_state.has_run_once or st.session_state.emotion_log:
    tab1, tab2 = st.tabs(["Detection", "Visualizer"])
else:
    tab1, = st.tabs(["Detection"])

# -------- TAB 1: DETECTION --------
with tab1:
    FRAME_WINDOW = st.image([])
    fps_display = st.empty()
    st.sidebar.header("Live Emotion Count")
    st.sidebar.markdown("---")
    sidebar_placeholders = {e: st.sidebar.empty() for e in emotion_labels}

    if run:
        camera = cv2.VideoCapture(0)
        frame_times = []

        while run:
            success, frame = camera.read()
            if not success:
                st.warning("Webcam not detected.")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = get_faces(gray)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            top_emotion = None

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                emotion, _ = get_prediction_probs(roi)
                st.session_state.emotion_counts[emotion] += 1
                top_emotion = emotion

                st.session_state.emotion_log.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "emotion": emotion
                })

                cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(frame_rgb, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Update emoji state if emotion changed
            if top_emotion and top_emotion in emojis and top_emotion != st.session_state.current_emoji:
                st.session_state.current_emoji = top_emotion
                st.session_state.last_emoji_change_time = time.time()

            # Animate emoji (pulse)
            if st.session_state.current_emoji in emojis:
                pulse = 0.6 + 0.4 * np.sin(time.time() * 3)
                base_emoji = emojis[st.session_state.current_emoji].resize((64, 64))
                enhancer = ImageEnhance.Brightness(base_emoji)
                faded_emoji = enhancer.enhance(pulse)
                frame_pil.paste(faded_emoji, (10, 10), faded_emoji)

            FRAME_WINDOW.image(np.array(frame_pil))

            now = time.time()
            frame_times.append(now)
            if len(frame_times) > 30:
                frame_times = frame_times[-30:]
            fps = len(frame_times) / (frame_times[-1] - frame_times[0] + 1e-6)
            st.session_state.fps_log.append(fps)
            fps_display.text("FPS: {:.2f}".format(fps))

            for e in emotion_labels:
                sidebar_placeholders[e].markdown(f"**{e}**: {st.session_state.emotion_counts[e]}")

            if snapshot:
                os.makedirs("snapshots", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"snapshots/snapshot_{timestamp}.png"
                cv2.imwrite(fname, cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR))
                st.success(f"Snapshot saved to {fname}")
                snapshot = False

        camera.release()
        st.session_state.has_run_once = True
# -------- TAB 2: VISUALIZER --------
if 'tab2' in locals():
    with tab2:
        st.subheader("Emotion Count Overview")
        colA, colB = st.columns(2)

        with colA:
            df_bar = pd.DataFrame.from_dict(
                st.session_state.emotion_counts, orient='index', columns=['Count']
            )
            fig_bar, ax_bar = plt.subplots()
            df_bar.plot(kind='bar', legend=False, ax=ax_bar, color='lightskyblue')
            ax_bar.set_title("Total Emotion Count")
            plt.xticks(rotation=45)
            st.pyplot(fig_bar)

        with colB:
            if st.session_state.emotion_log:
                df_log = pd.DataFrame(st.session_state.emotion_log)
                df_line = df_log.groupby(['timestamp', 'emotion']).size().unstack().fillna(0).cumsum()
                fig_line, ax_line = plt.subplots()
                df_line.plot(ax=ax_line, linewidth=2)
                ax_line.set_title("Emotion Over Time")
                ax_line.set_ylabel("Cumulative Count")
                plt.xticks(rotation=45)
                st.pyplot(fig_line)
            else:
                st.info("Waiting for detection to begin...")

        st.divider()
        st.subheader("Performance Metrics")
        total_detected = sum(st.session_state.emotion_counts.values())
        avg_fps = np.mean(st.session_state.fps_log) if st.session_state.fps_log else 0.0
        st.markdown(f"- **Total Detections:** {total_detected}")
        st.markdown(f"- **Average FPS:** {avg_fps:.2f}")

        if st.session_state.emotion_log:
            os.makedirs("logs", exist_ok=True)
            df = pd.DataFrame(st.session_state.emotion_log)
            log_file = f"logs/emotion_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(log_file, index=False)
            st.success(f"Log saved to {log_file}")
