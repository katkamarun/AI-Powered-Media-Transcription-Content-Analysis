import instructor
from pydantic import BaseModel, Field
from openai import OpenAI
import whisper
import streamlit as st
from pydub import AudioSegment
import numpy as np
import os
import tempfile

# Initialize the instructor client once
client = instructor.patch(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    ),
    mode=instructor.Mode.JSON,
)


class TitleSummary(BaseModel):
    title: str = Field(
        ...,
        description="A concise and descriptive title that captures the main theme of the transcript",
    )
    summary: str = Field(
        ...,
        description="A comprehensive summary that captures the key points and main ideas from the transcript",
    )


SYSTEM_PROMPT = """
You are an expert at analyzing transcripts and creating clear, concise titles and summaries.
Your task is to:
1. Create a title that captures the main theme or topic of the transcript
2. Generate a summary that highlights the key points and main ideas
Focus on accuracy, clarity, and conciseness in your output.
"""

st.title("Audio/Video Transcription and Analysis")
st.write("Upload an audio or video file to get a transcription, title, and summary")

# File uploader
uploaded_file = st.file_uploader(
    "Upload Audio/Video", type=["wav", "mp3", "m4a", "mp4"]
)


# Initialize Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


model = load_whisper_model()
st.text("Whisper Model Loaded")

if uploaded_file is not None:
    # Display the uploaded file
    st.sidebar.header("Original File")

    # Handle video files
    if uploaded_file.type.startswith("video/"):
        # Save video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name

        # Display video
        st.sidebar.video(video_path)

        # Extract audio from video
        audio = AudioSegment.from_file(video_path).set_channels(1).set_frame_rate(16000)

        # Clean up temporary file
        os.unlink(video_path)
    else:
        # Handle audio files
        st.sidebar.audio(uploaded_file)
        audio = (
            AudioSegment.from_file(uploaded_file).set_channels(1).set_frame_rate(16000)
        )

    if st.button("Process File"):
        try:
            with st.spinner("Processing file..."):
                # Convert audio to numpy array
                audio_np = (
                    np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
                )

                # Transcribe
                st.text("Transcribing audio...")
                result = model.transcribe(audio_np)
                transcription_text = result["text"]

                # Display transcription
                st.subheader("Transcription")
                st.text_area("Transcribed Text", transcription_text, height=200)

                # Generate title and summary
                st.text("Generating title and summary...")
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": transcription_text},
                ]

                response = client.chat.completions.create(
                    model="llama3.2",
                    messages=messages,
                    response_model=TitleSummary,
                )

                # Display results
                st.subheader("Analysis Results")
                st.markdown(f"**Title:** {response.title}")
                st.markdown(f"**Summary:** {response.summary}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error(
                "Please make sure Ollama is running and the llama3.2 model is available"
            )
