"""
Snowflake FAQ Agent Module

This module provides an LLM-powered agent that answers questions about
Snowflake concepts using the FAQ knowledge base and Cerebras inference.
"""

import os
from pathlib import Path
from cerebras.cloud.sdk import Cerebras

# Load FAQ content at module level
FAQ_PATH = Path(__file__).parent / "FAQ.md"

def load_faq_content() -> str:
    """Load the FAQ markdown content."""
    if FAQ_PATH.exists():
        return FAQ_PATH.read_text(encoding="utf-8")
    return ""

FAQ_CONTENT = load_faq_content()

SYSTEM_PROMPT = f"""You are a helpful Snowflake expert assistant. Your role is to answer questions about Snowflake data platform concepts, features, and terminology.

You have access to the following Snowflake FAQ knowledge base:

---
{FAQ_CONTENT}
---

Instructions:
1. Answer questions based on the FAQ content above.
2. If the question relates to a topic in the FAQ, provide a clear and concise answer.
3. If the question is about Snowflake but not covered in the FAQ, provide your best knowledge but mention that it may not be in the official glossary.
4. If the question is not related to Snowflake at all, politely redirect the user to ask Snowflake-related questions.
5. Be conversational and helpful.
6. When relevant, mention related concepts the user might want to learn about.
"""


class SnowflakeAgent:
    """
    An LLM-powered agent that answers Snowflake-related questions
    using Cerebras inference and the FAQ knowledge base.
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize the Snowflake agent.
        
        Args:
            api_key: Cerebras API key. If not provided, reads from CEREBRAS_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Cerebras API key is required. Pass it to the constructor or "
                "set the CEREBRAS_API_KEY environment variable."
            )
        self.client = Cerebras(api_key=self.api_key)
        self.model = "llama-3.3-70b"

    def answer(self, question: str) -> str:
        """
        Answer a question about Snowflake using the FAQ knowledge base.
        
        Args:
            question: The user's question about Snowflake.
            
        Returns:
            The agent's answer as a string.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
            )

            if (
                not hasattr(response, "choices")
                or len(response.choices) == 0
                or not response.choices[0].message
            ):
                return "I apologize, but I couldn't generate a response. Please try again."

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error communicating with the AI service: {str(e)}"


def get_snowflake_answer(question: str, api_key: str | None = None) -> str:
    """
    Convenience function to get an answer about Snowflake.
    
    Args:
        question: The user's question about Snowflake.
        api_key: Optional Cerebras API key.
        
    Returns:
        The agent's answer as a string.
    """
    agent = SnowflakeAgent(api_key=api_key)
    return agent.answer(question)

