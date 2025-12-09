import io
import os
import streamlit as st
from audiorecorder import audiorecorder
from faster_whisper import WhisperModel
from cerebras.cloud.sdk import Cerebras

# ---------------------------
# Streamlit page setup
# ---------------------------
st.set_page_config(page_title="Local Whisper + Cerebras", page_icon="🎙️")
st.title("🎙️ Local Whisper Transcriber with Cerebras Cleanup")

st.write(
    "Record audio below and transcribe it locally using Whisper (faster-whisper). "
    "Then optionally clean up the transcript using a Cerebras LLM."
)

# ---------------------------
# Cached resources
# ---------------------------
@st.cache_resource
def load_whisper_model():
    # choose: tiny, base, small, medium, large
    return WhisperModel("small", device="cpu")

@st.cache_resource
def get_cerebras_client():
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "CEREBRAS_API_KEY environment variable is not set. "
            "Please set it before running the app."
        )
    return Cerebras(api_key=api_key)

model = load_whisper_model()

# Lazy init of Cerebras client only if we actually need it
cerebras_client = None

# ---------------------------
# Helper: refine transcript with Cerebras
# ---------------------------

def refine_transcript_with_cerebras(raw_text: str) -> str:
    """
    Send the raw Whisper transcript to a Cerebras LLM
    and get back cleaned / corrected text.
    Handles errors gracefully.
    """
    global cerebras_client
    if cerebras_client is None:
        cerebras_client = get_cerebras_client()

    system_prompt = (
        "You are a transcription post-processor. "
        "You receive raw automatic speech recognition (ASR) output. "
        "Your job is to:\n"
        "- fix spelling and grammar\n"
        "- add punctuation and sentence breaks\n"
        "- keep the meaning unchanged\n"
        "Return only the cleaned text, no explanations."
    )

    try:
        response = cerebras_client.chat.completions.create(
            model="llama-3.3-70b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_text},
            ],
        )

        # Validate the response
        if (
            not hasattr(response, "choices")
            or len(response.choices) == 0
            or not response.choices[0].message
        ):
            st.error("⚠️ Cerebras returned an empty or invalid response.")
            return ""

        return response.choices[0].message.content.strip()

    except Exception as e:
        st.error("❌ Cerebras LLM failed to process the request.")
        st.exception(e)  # prints full traceback in Streamlit

        return ""


# ---------------------------
# Session state for transcript
# ---------------------------
if "raw_transcript" not in st.session_state:
    st.session_state.raw_transcript = ""

# ---------------------------
# Audio recorder
# ---------------------------
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

        st.session_state.raw_transcript = transcript_text

# ---------------------------
# Show raw transcript (if any)
# ---------------------------
if st.session_state.raw_transcript:
    st.subheader("Raw transcript (Whisper)")
    st.write(st.session_state.raw_transcript)

    # ---------------------------
    # Cerebras cleanup button
    # ---------------------------
    if st.button("✨ Clean & correct with Cerebras"):
        with st.spinner("Refining transcript with Cerebras..."):
            refined_text = refine_transcript_with_cerebras(
                st.session_state.raw_transcript
            )

        st.subheader("Refined transcript (Cerebras)")
        st.write(refined_text)
