import streamlit as st
import os

from components.sidebar import sidebar
from src import (
    api_calls,
    video_asr,
    video_ocr
)

OCR_SAMPLE_RATE = 2
API_MODEL_NAME = "gpt-3.5-turbo"
SUPPORTED_FILETYPES = ["mp4", "avi", "mov", "mkv", "wmv"]

ocr_helper = video_ocr.OCRHelper(filepath=None, sample_rate=OCR_SAMPLE_RATE)
text_extractor = video_asr.ASRHelper(audio_path=None)
chat_handler = api_calls.ChatHandler(API_MODEL_NAME)

def clear_submit():
    st.session_state["submit"] = False

st.set_page_config(page_title="Video Summarization", page_icon="🎥", layout="wide")
st.header("🎥Video Presentation Summarization")

sidebar()
if "OPENAI_API_KEY" not in st.session_state or not st.session_state["OPENAI_API_KEY"]:
    st.error("Please enter your OpenAI API key in the sidebar.")
else:
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=SUPPORTED_FILETYPES,
        on_change=clear_submit,
    )

    if uploaded_file is not None:
        filename, file_extension = os.path.splitext(uploaded_file.name)
        if file_extension[1:] in SUPPORTED_FILETYPES:
            video_path = f"{uploaded_file.name}"
            with open(video_path, "wb") as out_file:
                out_file.write(uploaded_file.getbuffer())
                submit_button = st.button("Start summarization")
            st.video(video_path)
        else:
            raise ValueError("File type not supported!")
        
        if submit_button:
            try:
                progress_bar = st.progress(0)
                # Perform OCR on the video
                with st.spinner("Performing OCR on video⏳"):
                    ocr_helper.filepath = video_path
                    ocr_text = ocr_helper.perform_video_ocr()

                st.markdown("✔️ OCR done")
                progress_bar.progress(0.33)

                # Perform ASR on the audio
                with st.spinner("Performing ASR on video⏳"):
                    text_extractor.audio_path = video_path
                    asr_text = text_extractor.extract_text_from_audio()

                progress_bar.progress(0.66)
                st.markdown("✔️ ASR done")

                # Get summary
                with st.spinner("Collecting summary⏳"):
                    summary = chat_handler.process_ocr_and_asr(ocr_text, asr_text)

                progress_bar.progress(1.0)
                st.markdown("✔️ Summary generated")

                # Show the summary
                st.markdown(f"## Summary")
                st.markdown(summary)

                # regenerate summary
                regenerate_button = st.button('Regenerate summary with a new video')
                if regenerate_button:
                    clear_submit()
                
            except Exception as e:

                st.error(f'Error processing video: {e}')
                st.warning('Please try again with a different video or check the video format.')
                clear_submit()

            finally:
                # Cleanup
                os.remove(video_path)
