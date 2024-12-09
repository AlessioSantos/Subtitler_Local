import streamlit as st
import whisper
import subprocess
import os
import tempfile
from pathlib import Path
from transformers import pipeline

st.markdown(
    """
    <h1 style='text-align: center; color: orange;'>Subtitler</h1>
    <p style='text-align: center; color: blue; font-size:18px;'>Built with passion By Zoe LAB 😘 (ver. 03).</p>
    """,
    unsafe_allow_html=True
)

model_name = "medium"

@st.cache_resource
def load_whisper_model(model_name):
    return whisper.load_model(model_name)

@st.cache_resource
def load_summarization_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def extract_audio_with_ffmpeg(uploaded_file, progress_callback=None):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_temp:
            video_temp.write(uploaded_file.getbuffer())
            video_temp_path = video_temp.name
        
        audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_output_path = audio_temp.name

        if progress_callback:
            progress_callback(20)

        command = [
            "ffmpeg", "-i", video_temp_path, "-q:a", "0", "-map", "a", audio_output_path, "-y"
        ]
        subprocess.run(command, check=True)

        if progress_callback:
            progress_callback(50)

        os.remove(video_temp_path)

        return audio_output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg command failed with error: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def generate_srt_with_whisper(audio_path, model_name="base", progress_callback=None):
    model = load_whisper_model(model_name)

    if progress_callback:
        progress_callback(70)

    result = model.transcribe(audio_path)

    srt_content = ""
    transcript = ""
    for i, segment in enumerate(result["segments"]):
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        transcript += f"{text} "

        srt_content += f"{i+1}\n"
        srt_content += f"{format_time(start)} --> {format_time(end)}\n"
        srt_content += f"{text}\n\n"

    if progress_callback:
        progress_callback(100)

    return srt_content, transcript

def summarize_text(transcript, max_length=130, min_length=30):
    summarizer = load_summarization_model()
    summary = summarizer(transcript, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]["summary_text"]

uploaded_file = st.sidebar.file_uploader("Load your video file", type=["mp4", "mkv", "avi"])
process_button = st.sidebar.button("Create Subtitles")
summarize_button = st.sidebar.button("Summarize Video")

video_placeholder = st.empty()

if "srt_content" not in st.session_state:
    st.session_state["srt_content"] = ""
    st.session_state["transcript"] = ""
    st.session_state["summary"] = ""

if uploaded_file:
    video_bytes = uploaded_file.read()
    video_placeholder.video(video_bytes)

if uploaded_file and process_button:
    progress_bar = st.progress(0)

    def update_progress(value):
        progress_bar.progress(value)

    with st.spinner("Processing the video file, please wait..."):
        try:
            audio_path = extract_audio_with_ffmpeg(uploaded_file, progress_callback=update_progress)
            srt_content, transcript = generate_srt_with_whisper(audio_path, model_name=model_name, progress_callback=update_progress)

            os.remove(audio_path)

            st.session_state["srt_content"] = srt_content
            st.session_state["transcript"] = transcript
            st.toast("Subtitles have been successfully created!", icon="🎉")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if uploaded_file and summarize_button:
    with st.spinner("Summarizing transcript..."):
        try:
            if not st.session_state.get("transcript"):
                audio_path = extract_audio_with_ffmpeg(uploaded_file)

                model = load_whisper_model(model_name)
                result = model.transcribe(audio_path)

                st.session_state["transcript"] = " ".join([seg["text"] for seg in result["segments"]])

                os.remove(audio_path)

            summary = summarize_text(st.session_state["transcript"])
            st.session_state["summary"] = summary
            st.toast("Summary successfully created!", icon="🎉")
        except Exception as e:
            st.error(f"An error occurred during summarization: {e}")

if st.session_state["srt_content"]:
    edited_srt_content = st.text_area(
        "Edit your subtitles below:",
        value=st.session_state["srt_content"],
        height=300,
        key="subtitle_editor"
    )

    st.session_state["srt_content"] = edited_srt_content

    st.download_button(
        label="Download Edited SRT",
        data=st.session_state["srt_content"],
        file_name="subtitles.srt",
        mime="text/srt"
    )

if st.session_state["summary"]:
    st.subheader("Summary:")
    st.text_area("Video Summary", st.session_state["summary"], height=150)

    st.download_button(
        label="Download Summary",
        data=st.session_state["summary"],
        file_name="summary.txt",
        mime="text/plain"
    )
