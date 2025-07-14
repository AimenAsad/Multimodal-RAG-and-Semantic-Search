from openai import OpenAI
from typing import List, Dict
import streamlit as st


class ResponseGenerator:
    def __init__(self):
        # Initialize Gemini client using your Streamlit secrets
        self.client = OpenAI(
            api_key=st.secrets["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    def generate_response(self, query: str, context: List[Dict], strategy: str = "cot") -> str:
        """Generate answer using specified prompting strategy"""
        # Build source context string
        context_str = "\n".join(
            f"Source {i+1} (Page {c['metadata']['page']}): {c['content']}"
            for i, c in enumerate(context)
        )

        # Prompt Engineering Strategies
        prompts = {
            "cot": f"""Analyze step-by-step using these sources:
{context_str}

Question: {query}

First, identify relevant facts from the sources.
Then, synthesize a comprehensive answer:""",
            "fewshot": f"""Examples:
Q: What was 2023 revenue?
A: $5.2M (Page 12 chart)

Q: {query}
A:"""
        }

        response = self.client.chat.completions.create(
            model="gemini-1.5-pro-latest",
            messages=[{"role": "user", "content": prompts[strategy]}],
            temperature=0.3
        )
        return response.choices[0].message.content
