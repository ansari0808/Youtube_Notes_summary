from typing import Dict, Any
from database.databaseInterface import DatabaseInterface
from utils.youtube_utils import YouTubeUtils

class YouTubeNotes:
    def __init__(self, database: DatabaseInterface):
        self.database = database
        self.utils = YouTubeUtils()

    def add_video(self, video_id: str) -> Dict[str, Any]:
        """Add a video transcript to the database."""
        transcript = self.utils.get_transcript(video_id)
        if not transcript:
            raise ValueError(f"Failed to fetch transcript for video {video_id}")

        embedding = self.utils.generate_embeddings(transcript)
        summary = self.utils.generate_summary(transcript)

        metadata = {"video_id": video_id, "summary": summary}
        self.database.add_embedding(
            id=video_id,
            embedding=embedding,
            document=transcript,
            metadata=metadata
        )

        return {"video_id": video_id, "summary": summary, "status": "success"}

    def ask_question(self, question: str) -> Dict[str, Any]:
        """Query the database for an answer."""
        query_embedding = self.utils.generate_embeddings(question)
        results = self.database.find_similar_embeddings(query_embedding)

        if not results or not results['documents']:
            return {"answer": "No relevant information found.", "context": None}

        context = " ".join(results['documents'][0])
        answer = self.utils.generate_answer(question, context)

        return {
            "answer": answer,
            "context": results['documents'][0],
            "video_ids": results['ids'][0]
        }