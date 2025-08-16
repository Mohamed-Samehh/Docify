from groq import Groq
from typing import List, Generator
from langchain.schema import Document

class ChatbotService:
    def __init__(self, api_key: str = None):
        self.client = Groq(api_key=api_key)
        self.model = "moonshotai/kimi-k2-instruct"
        self.vision_model = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    def summarize_document(self, chunks: List[Document]) -> str:
        """Generate comprehensive document summary for long documents"""
        
        # Check if we have images
        has_images = any(chunk.metadata.get("type") == "image" for chunk in chunks)
        
        if has_images:
            return self._summarize_image_document(chunks)
        
        # If document is small, use original approach
        full_text = "\n".join([chunk.page_content for chunk in chunks])
        if len(full_text) <= 8000:
            return self._summarize_short_document(full_text)
        
        # For long documents, use chunk-based summarization
        return self._summarize_long_document(chunks)
    
    def _summarize_short_document(self, full_text: str) -> str:
        """Original summarization for short documents"""
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
    
    def _summarize_long_document(self, chunks: List[Document]) -> str:
        """Enhanced summarization for long documents using chunk-based approach"""
        
        # Step 1: Create summaries for groups of chunks
        chunk_summaries = []
        chunk_size = 5  # Process 5 chunks at a time
        
        for i in range(0, len(chunks), chunk_size):
            chunk_group = chunks[i:i + chunk_size]
            group_text = "\n".join([chunk.page_content for chunk in chunk_group])
            
            # Limit each group to reasonable size
            if len(group_text) > 6000:
                group_text = group_text[:6000] + "..."
            
            prompt = f"""Please provide a concise summary of this section of the document:

{group_text}

Focus on key concepts, main topics, and important details. Summary:"""
            
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_completion_tokens=512,
                    stream=False
                )
                
                chunk_summaries.append(f"Section {i//chunk_size + 1}: {completion.choices[0].message.content}")
            except Exception as e:
                chunk_summaries.append(f"Section {i//chunk_size + 1}: [Error summarizing this section: {str(e)}]")
        
        # Step 2: Create final comprehensive summary from chunk summaries
        all_summaries = "\n\n".join(chunk_summaries)
        
        final_prompt = f"""Based on the following section summaries of a document, create a comprehensive overall summary that covers all the main topics and key points:

{all_summaries}

Please provide a well-structured summary that:
1. Covers all major topics mentioned across sections
2. Maintains logical flow and organization
3. Highlights key concepts and important details
4. Is comprehensive yet concise

Comprehensive Summary:"""
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.3,
                max_completion_tokens=2048,
                stream=False
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            # Fallback: return the section summaries if final summarization fails
            return f"Document Summary (Section-based):\n\n{all_summaries}"
    
    def _summarize_image_document(self, chunks: List[Document]) -> str:
        """Summarize document containing images using vision model"""
        for chunk in chunks:
            if chunk.metadata.get("type") == "image":
                # Extract base64 image data
                content = chunk.page_content
                if "[IMAGE_DATA]" in content and "[/IMAGE_DATA]" in content:
                    image_data = content.split("[IMAGE_DATA]")[1].split("[/IMAGE_DATA]")[0]
                    
                    # Use vision model to analyze the image
                    completion = self.client.chat.completions.create(
                        model=self.vision_model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Please provide a comprehensive summary and analysis of this image. Describe what you see, any text content, important details, and overall context."
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_data}"
                                        }
                                    }
                                ]
                            }
                        ],
                        temperature=0.3,
                        max_completion_tokens=1024,
                        stream=False
                    )
                    
                    return completion.choices[0].message.content
        
        return "No valid image found for analysis."
    
    def answer_question(self, question: str, context_docs: List[Document]) -> Generator[str, None, None]:
        """Answer question based on context documents with streaming"""
        
        # Check if we have images in the context
        has_images = any(doc.metadata.get("type") == "image" for doc in context_docs)
        
        if has_images:
            yield from self._answer_question_with_images(question, context_docs)
        else:
            yield from self._answer_question_text_only(question, context_docs)
    
    def _answer_question_text_only(self, question: str, context_docs: List[Document]) -> Generator[str, None, None]:
        """Answer question for text-only documents"""
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
    
    def _answer_question_with_images(self, question: str, context_docs: List[Document]) -> Generator[str, None, None]:
        """Answer question when images are present in context"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Please answer this question based on the provided image(s): {question}"
                    }
                ]
            }
        ]
        
        # Add images to the message
        for doc in context_docs:
            if doc.metadata.get("type") == "image":
                content = doc.page_content
                if "[IMAGE_DATA]" in content and "[/IMAGE_DATA]" in content:
                    image_data = content.split("[IMAGE_DATA]")[1].split("[/IMAGE_DATA]")[0]
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    })
            else:
                # Add text context as well
                messages[0]["content"].append({
                    "type": "text",
                    "text": f"Additional context: {doc.page_content}"
                })
        
        completion = self.client.chat.completions.create(
            model=self.vision_model,
            messages=messages,
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