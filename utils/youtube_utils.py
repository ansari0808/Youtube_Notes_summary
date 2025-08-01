from typing import Optional, List, Dict, Any
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi
import ollama

class YouTubeUtils:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_transcript(self, video_id: str) -> Optional[str]:
        """Fetch YouTube transcript for a given video ID."""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([entry["text"] for entry in transcript])
        except Exception as e:
            print(f"Error fetching transcript: {e}")
            return None

    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for a given text."""
        return self.embedding_model.encode(text).tolist()

    def generate_summary(self, text: str, model: str = "deepseek-r1:1.5b") -> str:
        """Generate a summary using Ollama."""
        response = ollama.generate(
            model=model,
            prompt=f"Summarize the following text concisely:\n{text}",
            options={"max_tokens": 150, "temperature": 0.7},
        )
        return response["response"]

    def generate_answer(self, question: str, context: str) -> str:
        """Generate an answer ONLY from the provided context. Returns 'I don't know' for out-of-context questions."""
        strict_prompt = f"""
        [SYSTEM]
        Role: You are a fact-based QA assistant that ONLY answers questions using the provided context.
        Rules:
        1. If the answer isn't explicitly in the context, respond: "I don't have information about this in the video."
        2. Never use outside knowledge.
        3. If unsure, say you don't know.
        4. Keep answers concise (1-2 sentences max).

        [CONTEXT]
        {context}

        [QUESTION]
        {question}

        [ANSWER]
        """
    
        response = ollama.generate(
            model="deepseek-r1:1.5b",
            prompt=strict_prompt,
            options={
                "max_tokens": 150,  # Shorter to prevent rambling
                "temperature": 0.3, # Lower = more deterministic
                "top_p": 0.85,     # Focus on high-probability words
                "repeat_penalty": 1.2,
            },
        )
        answer = response["response"].strip()
    
        # Secondary validation to catch hallucinations
        required_phrases = ["don't know", "don't have information", "not mentioned"]
        if not any(phrase in answer.lower() for phrase in required_phrases):
            if not self._is_answer_in_context(answer, context):
                answer = "I don't have information about this in the video."
    
        return answer

    def _is_answer_in_context(self, answer: str, context: str) -> bool:
        """Validate if key terms from the answer exist in the context."""
        answer_terms = set(answer.lower().split())
        context_terms = set(context.lower().split())
        return len(answer_terms & context_terms) >= 2  # At least 2 overlapping keywords