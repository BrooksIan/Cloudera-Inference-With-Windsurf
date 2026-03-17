#windsurf_agent/ClouderaLLMClient.py
from openai import OpenAI
from typing import Dict, List, Optional, Generator
import os
from dotenv import load_dotenv
from pathlib import Path


class ClouderaLLMClient:
    def __init__(self):
        # Ensure we're only using Cloudera endpoints
        self.base_url = os.getenv("WINDSURF_LLM_BASE_URL")
        if not self.base_url or "cloudera.site" not in self.base_url:
            raise ValueError("Only Cloudera ML endpoints are allowed")

        self.api_key = os.getenv("WINDSURF_LLM_API_KEY")
        self.model = os.getenv("WINDSURF_LLM_MODEL", "goes---nemotron-v1-5-49b-throughput")

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers={
                "X-Requested-By": "cascade-agent",
                "X-XSRF-Header": "true",
            }
        )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = True
    ) -> Generator[str, None, None]:
        """
        Generate a chat completion.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model ID to use (defaults to instance model)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response

        Yields:
            Chunks of the generated text
        """
        model = model or self.model
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=stream
            )

            if stream:
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
            else:
                yield response.choices[0].message.content

        except Exception as e:
            raise Exception(f"Error generating completion: {str(e)}")