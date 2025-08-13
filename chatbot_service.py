from groq import Groq
from typing import List, Generator
from langchain.schema import Document

class ChatbotService:
    def __init__(self, api_key: str = None):
        self.client = Groq(api_key=api_key)
        self.model = "moonshotai/kimi-k2-instruct"
    
    def summarize_document(self, chunks: List[Document]) -> str:
        """Generate document summary"""
        # Combine chunks for summary
        full_text = "\n".join([chunk.page_content for chunk in chunks])
        
        # Truncate if too long (keep first 8000 chars for summary)
        if len(full_text) > 8000:
            full_text = full_text[:8000] + "..."
        
        prompt = f"""Please provide a comprehensive summary of the following document:

{full_text}

Summary:"""
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=1024,
            stream=False
        )
        
        return completion.choices[0].message.content
    
    def answer_question(self, question: str, context_docs: List[Document]) -> Generator[str, None, None]:
        """Answer question based on context documents with streaming"""
        context = "\n".join([doc.page_content for doc in context_docs])
        
        prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_completion_tokens=2048,
            stream=True
        )
        
        for chunk in completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def chat_conversation(self, messages: List[dict], context_docs: List[Document] = None) -> Generator[str, None, None]:
        """Handle multi-turn conversation with optional context"""
        if context_docs:
            context = "\n".join([doc.page_content for doc in context_docs])
            system_message = {
                "role": "system",
                "content": f"You are a helpful assistant. Use this context to answer questions: {context}"
            }
            messages = [system_message] + messages
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.6,
            max_completion_tokens=2048,
            stream=True
        )
        
        for chunk in completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content