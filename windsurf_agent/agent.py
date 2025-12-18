import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

from .config import Config
from .embedding_client import WindsurfEmbeddingClient
from .llm_client import WindsurfLLMClient
from .vector_store import SimpleVectorStore, Document
from .exceptions import WindsurfError, EmbeddingError, LLMError, VectorStoreError

logger = logging.getLogger(__name__)

class WindsurfAgent:
    """Main agent class for interacting with Windsurf's AI models."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the Windsurf agent.
        
        Args:
            config: Optional configuration. If not provided, will be loaded from environment.
        """
        self.config = config or Config.from_env()
        self._setup_clients()

    def _make_request(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Make a request to the LLM with the given messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the LLM client
            
        Returns:
            The generated response text
            
        Raises:
            LLMError: If the request fails
        """
        try:
            # Get the model name from the LLM config
            model = self.config.llm.model if hasattr(self.config.llm, 'model') else None
            
            # Delegate the chat completion to the LLM client
            response = self.llm_client.chat_complete(
                messages=messages,
                model=model,
                **kwargs
            )
            return response
        except Exception as e:
            logger.error(f"LLM request failed: {str(e)}")
            raise LLMError(f"Failed to make LLM request: {str(e)}") from e

    def _setup_clients(self):
        """Set up the required clients."""
        try:
            self.embedding_client = WindsurfEmbeddingClient(self.config.embedding)
            self.llm_client = WindsurfLLMClient(self.config.llm)
            self.vector_store = SimpleVectorStore(self.config.vector_store)
            logger.info("WindsurfAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WindsurfAgent: {str(e)}")
            raise

    def embed(self, text: str) -> List[float]:
        """Get embedding for a single text.
        
        Args:
            text: The text to embed.
            
        Returns:
            A list of floats representing the embedding.
        """
        return self.embedding_client.get_embedding(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            A list of embeddings, one for each input text.
        """
        return self.embedding_client.get_embeddings(texts)

    def add_to_knowledge_base(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None
    ) -> List[str]:
        """Add documents to the knowledge base.
        
        Args:
            texts: List of document texts to add.
            metadatas: Optional list of metadata dictionaries.
            
        Returns:
            List of document IDs.
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)
            
        if len(texts) != len(metadatas):
            raise ValueError("Number of texts must match number of metadata dictionaries")
            
        try:
            # Get embeddings for all texts
            embeddings = self.embed_batch(texts)
            # Add to vector store
            return self.vector_store.add_documents(texts, embeddings, metadatas)
        except Exception as e:
            logger.error(f"Failed to add documents to knowledge base: {str(e)}")
            raise VectorStoreError(f"Failed to add documents to knowledge base: {str(e)}") from e

    def search(
        self,
        query: str,
        k: int = 5,
        filter_func: Optional[callable] = None
    ) -> List[Tuple[Document, float]]:
        """Search the knowledge base for similar documents.
        
        Args:
            query: The search query.
            k: Number of results to return.
            filter_func: Optional function to filter documents.
            
        Returns:
            List of (document, similarity_score) tuples.
        """
        try:
            # Get embedding for the query
            query_embedding = self.embed(query)
            # Search the vector store
            return self.vector_store.similarity_search(query_embedding, k=k, filter_func=filter_func)
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise VectorStoreError(f"Search failed: {str(e)}") from e

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text based on a prompt.
        
        Args:
            prompt: The prompt to generate text from.
            temperature: Controls randomness (0.0 to 1.0).
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            The generated text.
        """
        try:
            return self.llm_client.complete(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise LLMError(f"Text generation failed: {str(e)}") from e

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a chat completion.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the LLM client
            
        Returns:
            The generated response text
            
        Raises:
            LLMError: If the request fails
        """
        try:
            # Get LLM config with fallbacks
            llm_config = self.config.llm
            model = getattr(llm_config, 'model', 'gpt-4')
            temperature = kwargs.get("temperature", getattr(llm_config, 'temperature', 0.7))
            max_tokens = kwargs.get("max_tokens", getattr(llm_config, 'max_tokens', 2048))
            
            # Call the LLM client's chat method (not chat_complete)
            response = self.llm_client.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
            )
            
            # The LLM client's chat method already returns the response text
            return response
            
        except Exception as e:
            logger.error(f"Chat completion failed: {str(e)}")
            raise LLMError(f"Chat completion failed: {str(e)}") from e
            
    def rag_query(
        self,
        query: str,
        k: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform a RAG (Retrieval Augmented Generation) query.
        
        Args:
            query: The query to search for.
            k: Number of relevant documents to retrieve.
            temperature: Controls randomness (0.0 to 1.0).
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            A dictionary containing:
            - answer: The generated answer
            - sources: List of source documents with scores
        """
        try:
            # Search for relevant documents
            results = self.search(query, k=k)
            
            if not results:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": []
                }
            
            # Format the context
            context = "\n\n".join([
                f"Document {i+1} (Score: {score:.3f}):\n{doc.text}"
                for i, (doc, score) in enumerate(results)
            ])
            
            # Generate the prompt
            prompt = f"""You are a helpful assistant. Use the following context to answer the question at the end. If you don't know the answer, just say you don't know, don't try to make up an answer.

Context:
{context}

Question: {query}

Answer:"""
            
            # Generate the answer
            answer = self.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Prepare sources
            sources = [{
                "text": doc.text,
                "score": float(score),
                "metadata": doc.metadata
            } for doc, score in results]
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"RAG query failed: {str(e)}")
            raise WindsurfError(f"RAG query failed: {str(e)}") from e

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'embedding_client') and hasattr(self.embedding_client, 'session'):
            self.embedding_client.session.close()
        if hasattr(self, 'llm_client') and hasattr(self.llm_client, 'session'):
            self.llm_client.session.close()