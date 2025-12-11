import io
import os
import streamlit as st
from audiorecorder import audiorecorder
from faster_whisper import WhisperModel
from snowflake_agent import SnowflakeAgent

# ---------------------------
# Streamlit page setup
# ---------------------------
st.set_page_config(page_title="Snowflake Voice Assistant", page_icon="❄️")
st.title("❄️ Snowflake Voice Assistant")

st.write(
    "Ask questions about Snowflake using your voice! "
    "Record your question, get it transcribed, and receive an AI-powered answer "
    "based on the Snowflake FAQ knowledge base."
)

# ---------------------------
# Cached resources
# ---------------------------
@st.cache_resource
def load_whisper_model():
    # choose: tiny, base, small, medium, large
    return WhisperModel("small", device="cpu")

@st.cache_resource
def get_snowflake_agent():
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "CEREBRAS_API_KEY environment variable is not set. "
            "Please set it before running the app."
        )
    return SnowflakeAgent(api_key=api_key)

model = load_whisper_model()

# ---------------------------
# Session state
# ---------------------------
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""

# ---------------------------
# Audio recorder
# ---------------------------
st.subheader("🎙️ Step 1: Record Your Question")
audio = audiorecorder("🔴 Click to start / stop recording", "⏺️ Recording...")

if len(audio) > 0:
    st.audio(audio.export().read(), format="audio/wav")
    st.success("Recording captured! Click the button below to transcribe.")

    if st.button("📝 Transcribe with Whisper"):
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

        st.session_state.transcript = transcript_text
        st.session_state.answer = ""  # Clear previous answer

# ---------------------------
# Show transcript and get answer
# ---------------------------
if st.session_state.transcript:
    st.subheader("📄 Step 2: Your Transcribed Question")
    
    # Allow editing the transcript
    edited_transcript = st.text_area(
        "Edit your question if needed:",
        value=st.session_state.transcript,
        height=100,
        key="transcript_editor"
    )
    
    st.subheader("🤖 Step 3: Get Your Answer")
    
    if st.button("❄️ Ask Snowflake Agent"):
        with st.spinner("Consulting the Snowflake knowledge base..."):
            try:
                agent = get_snowflake_agent()
                answer = agent.answer(edited_transcript)
                st.session_state.answer = answer
            except Exception as e:
                st.error(f"Error getting answer: {str(e)}")

# ---------------------------
# Display the answer
# ---------------------------
if st.session_state.answer:
    st.subheader("💬 Answer")
    st.markdown(st.session_state.answer)
    
    # Option to ask follow-up
    st.divider()
    st.caption("Want to ask another question? Record a new audio above or type below:")
    
    followup = st.text_input("Type a follow-up question:", key="followup_input")
    if followup and st.button("Ask Follow-up"):
        with st.spinner("Getting answer..."):
            try:
                agent = get_snowflake_agent()
                answer = agent.answer(followup)
                st.session_state.answer = answer
                st.rerun()
            except Exception as e:
                st.error(f"Error getting answer: {str(e)}")

# ---------------------------
# Sidebar with info
# ---------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This app uses:\n"
        "- **Whisper** for local speech-to-text\n"
        "- **Cerebras LLM** for AI-powered answers\n"
        "- **Snowflake FAQ** as the knowledge base"
    )
    
    st.divider()
    st.header("💡 Example Questions")
    st.write(
        "- What is a virtual warehouse?\n"
        "- Explain Time Travel in Snowflake\n"
        "- What's the difference between streams and tasks?\n"
        "- How does zero-copy cloning work?\n"
        "- What is Snowflake Cortex AI?"
    )
