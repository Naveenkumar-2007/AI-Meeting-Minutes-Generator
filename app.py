import os
import streamlit as st
from tempfile import NamedTemporaryFile
import time

from src.pipeline.inference_pipeline import MeetingMinutesPipeline
from src.utils.helpers import create_download_link, get_example_audio

# Page configuration
st.set_page_config(
    page_title="AI Meeting Minutes Generator",
    page_icon="📝",
    layout="wide"
)

def main():
    # Title and description
    st.title("📝 AI Meeting Minutes Generator")
    st.markdown("""
    Upload a meeting recording to generate:
    * ✅ Detailed transcript
    * 📋 Summarized meeting minutes
    * ✓ Action items extraction
    * 🎭 Tone/sentiment analysis
    """)
    
    # Sidebar with options
    st.sidebar.title("Settings")
    summarization_model = st.sidebar.selectbox(
        "Summarization Model", 
        ["google/pegasus-xsum", "facebook/bart-large-cnn"]
    )
    
    whisper_model = st.sidebar.selectbox(
        "Transcription Model", 
        ["tiny", "base", "small", "medium", "large"]
    )
    
    extract_action_items = st.sidebar.checkbox("Extract Action Items", value=True)
    analyze_sentiment = st.sidebar.checkbox("Analyze Tone & Sentiment", value=True)
    
    # File upload
    st.subheader("Upload Meeting Recording")
    audio_file = st.file_uploader("Upload an audio file (MP3, WAV, M4A)", 
                                type=["mp3", "wav", "m4a"])
    
    # Example audio option
    use_example = st.checkbox("Or use an example recording")
    if use_example:
        audio_file = get_example_audio()
        st.audio(audio_file, format="audio/mp3")
    
    # Process button
    if st.button("Generate Minutes", type="primary") and (audio_file is not None):
        with st.spinner("Processing your recording... This may take a few minutes."):
            # Save uploaded file temporarily
            if audio_file is not None:
                with NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    audio_path = tmp_file.name
                    
                # Initialize pipeline
                pipeline = MeetingMinutesPipeline(
                    whisper_model_size=whisper_model,
                    summarizer_model_name=summarization_model,
                    extract_action_items=extract_action_items,
                    analyze_sentiment=analyze_sentiment
                )
                
                # Process the audio
                result = pipeline.process(audio_path)
                
                # Clean up temporary file
                os.unlink(audio_path)
                
                # Show results in tabs
                tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Transcript", "Action Items", "Sentiment Analysis"])
                
                with tab1:
                    st.subheader("Meeting Summary")
                    st.markdown(result['summary'])
                    st.download_button(
                        "Download Summary",
                        result['summary'],
                        file_name="meeting_summary.txt"
                    )
                
                with tab2:
                    st.subheader("Full Transcript")
                    st.markdown(result['transcript'])
                    st.download_button(
                        "Download Transcript",
                        result['transcript'],
                        file_name="meeting_transcript.txt"
                    )
                
                with tab3:
                    st.subheader("Action Items")
                    if extract_action_items:
                        for idx, item in enumerate(result['action_items']):
                            st.markdown(f"**{idx+1}.** {item}")
                        st.download_button(
                            "Download Action Items",
                            "\n".join([f"{idx+1}. {item}" for idx, item in enumerate(result['action_items'])]),
                            file_name="action_items.txt"
                        )
                    else:
                        st.info("Action item extraction was disabled")
                
                with tab4:
                    st.subheader("Sentiment Analysis")
                    if analyze_sentiment:
                        # Display overall sentiment
                        st.markdown(f"**Overall Meeting Tone:** {result['sentiment']['overall']}")
                        
                        # Display sentiment chart
                        st.subheader("Sentiment Timeline")
                        st.line_chart(result['sentiment']['timeline'])
                        
                        # Display key emotional moments
                        st.subheader("Key Emotional Moments")
                        for moment in result['sentiment']['key_moments']:
                            st.markdown(f"- **{moment['emotion']}** ({moment['time']}): {moment['text']}")
                    else:
                        st.info("Sentiment analysis was disabled")
                
                # Final download option for complete report
                st.subheader("Complete Report")
                st.download_button(
                    "Download Complete Meeting Report (PDF)",
                    "Report generation placeholder", # In a real implementation, generate PDF here
                    file_name="meeting_report.pdf"
                )

if __name__ == "__main__":
    main()