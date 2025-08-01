import streamlit as st
from youtube_notes import YouTubeNotes
from database.chroma import ChromaDB

def main():
    st.set_page_config(
        page_title="YouTube Notes RAG System", 
        page_icon="🎥", 
        layout="wide"
    )

    st.title("🎥 YouTube Notes RAG System")
    st.markdown("""
        This system allows you to:
        - Fetch YouTube video transcripts
        - Generate embeddings and store them
        - Query the system for answers
    """)

    # Initialize components
    database = ChromaDB()
    notes_system = YouTubeNotes(database)

    # Sidebar for video processing
    with st.sidebar:
        st.header("Process a Video")
        video_id = st.text_input("Enter YouTube Video ID:", placeholder="e.g., pNJUyol15Jw")
        
        if st.button("Process Video"):
            if video_id:
                with st.spinner("Processing video..."):
                    try:
                        result = notes_system.add_video(video_id)
                        st.success("Video processed successfully!")
                        st.subheader("Summary")
                        st.write(result["summary"])
                    except Exception as e:
                        st.error(f"Error processing video: {e}")
            else:
                st.warning("Please enter a valid YouTube Video ID.")

    # Main area for querying
    st.header("Ask Questions")
    question = st.text_input("Enter your question:", placeholder="How is data ingestion done in Elasticsearch?")
    
    if st.button("Get Answer"):
        if question:
            with st.spinner("Searching for answers..."):
                result = notes_system.ask_question(question)
                st.subheader("Answer")
                st.write(result["answer"])
                
                if result["context"]:
                    with st.expander("See relevant context"):
                        st.write(result["context"][:5])
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()