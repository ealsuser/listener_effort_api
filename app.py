import streamlit as st
import requests # type: ignore
import base64, json

# Page config & background
st.set_page_config(page_title="Listener Effort Prediction", layout="centered")

BASE_URL = "http://127.0.0.1:8000"

st.title("🔊 Listener Effort Prediction")

# 1) up to 3 recorders
audio_files = []
for i in range(3):
    audio = st.audio_input(f"Record voice message #{i+1}", key=f"audio_{i}")
    if audio is not None:
        st.audio(audio)
        st.write(f"Audio #{i+1} size: {audio.size} bytes")
    audio_files.append(audio)

# only proceed if ≥1 recording
recordings = [a for a in audio_files if a is not None]

if recordings:
    if st.button("Analyze Listener Effort"):
        with st.spinner("Processing audios..."):
            payload = {
                "files": {
                    f"{i+1}__{rec.name}": {
                        "wav_b64": base64.b64encode(rec.read()).decode(),
                        "metadata": {"metadata": f"File {i+1}"}
                    }
                    for i, rec in enumerate(recordings)
                }
            }
            resp = requests.post(
                f"{BASE_URL}/predict_from_bytes",
                json=payload,
                timeout=120
            )
            resp.raise_for_status()
            results = resp.json()

        # 1) Big rounded prediction
        pred = results.get("prediction", results.get("listener_effort"))
        st.markdown(f"<h1>Predicted Listener Effort: {pred:.3f}</h1>", unsafe_allow_html=True)

        # 2) Features section
        st.subheader("Features")
        for k, v in results.get("features", {}).items():
            st.write(f"**{k}**: {v}")

        # 3) Transcriptions section
        st.subheader("Transcriptions")
        for i, t in enumerate(results.get("transcripts", {}).values(), start=1):
            text = t.get("whisper_result", {}).get("text", "")
            st.write(f"Audio {i}: {text}")
        
        st.write(results)
else:
    st.info("Record at least one audio clip to enable Listener Effort prediction.")
