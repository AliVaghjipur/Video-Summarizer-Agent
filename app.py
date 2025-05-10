import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai

import time
from pathlib import Path

import tempfile
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from urllib.parse import urlparse, parse_qs

#Environment Variables from .env file
from dotenv import load_dotenv
load_dotenv()

import os

API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
    
# Streamlit Page Configuration
st.set_page_config(
    page_title="Agent - Video Summarizer",
    page_icon=":movie_camera:",
    layout="wide",
)

st.title("Phidata Video Ai Summarizer Agent")
st.header("Using Gemini AI")

@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[
            DuckDuckGo(),
        ],
        markdown=True,
    )
    

##Initiailizing the agent
multimodal_agent = initialize_agent()

tabs = st.tabs(["Upload Video", "Youtube Link"])


# ------------- Tab 1: Upload Video -------------
with tabs[0]:
    st.subheader("Upload a Video File (Max 200MB)")

    # FIle uploader
    video_file = st.file_uploader("Upload a video file, max 200MB", type=['mp4', 'mov', 'avi'], help="Upload a video file")

    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name
            
        st.video(video_path, format="video/mp4", start_time=0)
        
        user_query = st.text_area("What is the video about? What are the key points? What are the main topics discussed?", 
                                placeholder="Ask a question about the video. The AI agent will analyze the video and gather information to answer your question.",
                                help="Provide specific questions or insights you want from the video.")

        if st.button("Analyze Video", key="analyze_video_button"):
            if not user_query:
                st.warning("Please enter a question about the video.")
            else:
                try:    
                    with st.spinner("Analyzing video and gathering insights..."):
                        # Upload the video file to Gemini
                        processed_video = upload_file(video_path)
                        while processed_video.state.name == "PROCESSING":
                            time.sleep(1)
                            processed_video = get_file(processed_video.name)
                            
                        analysis_prompt = (
                            f"""
                            Analyze the uploaded video for content and context. Respond to the following query using video insights and supplementary web research 
                            {user_query}"
                            
                            Provide a detailed, user friendly, actionable response. 
                            """
                        )
                        
                        #AI Agent processing
                        response = multimodal_agent.run(analysis_prompt, videos=[processed_video])
                        
                    # Display result
                    st.subheader("Analysis Result")
                    st.markdown(response.content)
                    
                except Exception as error:
                    st.error(f"An error occurred: {error}")
                    
                finally:
                    #Clean up the temporary video file
                    Path(video_path).unlink(missing_ok=True)
                    
    # else:
    #     st.info("Please upload a video file to get started.")
    
    
# ------------- Tab 2: Youtube Link -------------
with tabs[1]:
    st.subheader("Paste a YouTube Video URL")
    youtube_url = st.text_input("YouTube URL")

    if youtube_url:
        def extract_video_id(url):
            parsed_url = urlparse(url)
            if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
                return parse_qs(parsed_url.query).get("v", [None])[0]
            elif parsed_url.hostname == 'youtu.be':
                return parsed_url.path[1:]
            return None

        video_id = extract_video_id(youtube_url)

        if video_id:
            try:
                transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = " ".join([entry['text'] for entry in transcript_data])

                st.success("Transcript successfully retrieved!")
                user_query_yt = st.text_area("What would you like to know from this video?", 
                                            placeholder="Ask a question about the YouTube video content.")

                if st.button("Analyze YouTube Video", key="analyze_youtube"):
                    if not user_query_yt:
                        st.warning("Please enter a query.")
                    else:
                        with st.spinner("Analyzing YouTube transcript using Gemini..."):
                            prompt = f"""
                            The following transcript is from a YouTube video. Analyze its content and respond to the query:
                            Transcript:
                            {transcript_text}

                            Query:
                            {user_query_yt}

                            Provide a clear, structured, insightful answer.
                            """
                            response = multimodal_agent.run(prompt)

                        st.subheader("Analysis Result")
                        st.markdown(response.content)

                        # Immediately delete transcript data  to save storage
                        del transcript_text

            except TranscriptsDisabled:
                st.error("Transcripts are disabled or unavailable for this video.")
            except Exception as e:
                st.error(f"An error occurred while retrieving transcript: {e}")
    
#Customize text and height
st.markdown(
    """
    <style>
    .stTextArea {
        height: 300px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)