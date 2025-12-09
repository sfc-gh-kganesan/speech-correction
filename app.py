import io
import streamlit as st
from audiorecorder import audiorecorder
from faster_whisper import WhisperModel
from pydub import AudioSegment

# Streamlit setup
st.set_page_config(page_title="Local Whisper Transcriber", page_icon="🎙️")
st.title("🎙️ Local Whisper Transcriber (No API Key Needed)")

st.write("Record audio below and transcribe it locally using Whisper (faster-whisper).")

# Load Whisper model (lazy loading)
@st.cache_resource
def load_whisper_model():
    return WhisperModel("small", device="cpu")  # choose: tiny, base, small, medium, large
model = load_whisper_model()

# Audio recorder
audio = audiorecorder("🔴 Click to start / stop recording", "⏺️ Recording...")

if len(audio) > 0:
    st.audio(audio.export().read(), format="audio/wav")
    st.success("Recording captured! Click the button below to transcribe.")

    if st.button("📝 Transcribe locally with Whisper"):
        with st.spinner("Transcribing... this may take a few seconds"):
            # Export audio to WAV bytes
            wav_bytes_io = io.BytesIO()
            audio.export(wav_bytes_io, format="wav")
            wav_bytes_io.seek(0)

            # Save to a temp file (Whisper needs a path)
            with open("temp.wav", "wb") as f:
                f.write(wav_bytes_io.read())

            # Run Whisper locally
            segments, info = model.transcribe("temp.wav")

            # Collect transcript text
            transcript_text = " ".join([seg.text for seg in segments])

        st.subheader("Transcript")
        st.write(transcript_text)
