from typing import List, Dict
from langchain_openai import ChatOpenAI

from llm_agent.utils.dataset_logger import DatasetLogger
from llm_agent.utils.prompts import AgentPrompts
import re

class LLMReasoning:
    """Handles LLM-based reasoning tasks: Classification, Refinement, Summarization, etc."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.dataset_logger = DatasetLogger()

    def summarize_history(self, chat_history: List[Dict]) -> List[Dict]:
        """Summarize history if too long to save tokens"""
        if not chat_history or len(chat_history) < 5:
            return chat_history

        # Keep last 2 messages intact, summarize the rest
        to_summarize = chat_history[:-2]
        recent = chat_history[-2:]

        text_block = "\\n".join([f"{msg['role']}: {msg['content']}" for msg in to_summarize])

        prompt = AgentPrompts.SUMMARIZE_HISTORY.format(history=text_block)
        response = self.llm.invoke([("human", prompt)])
        summary_text = response.content

        new_history = [
            {"role": "system", "content": f"Previous conversation summary: {summary_text}"}
        ] + recent

        return new_history

    def classify_query(self, query: str) -> str:
        """Classify query into categories"""
        messages = [
            ("system", AgentPrompts.CLASSIFICATION_SYSTEM),
            ("human", query),
        ]
        response = self.llm.invoke(messages)
        return response.content.strip().upper()

    def refine_semantic_query(self, query: str) -> str:
        """Refine query for vector search"""
        prompt = AgentPrompts.REFINE_QUERY.format(query=query)
        response = self.llm.invoke([("human", prompt)])
        refined_query = response.content.strip()

        # Log for refinement fine-tuning
        # We extract the static instruction part of the prompt as the System Message
        system_instruction = AgentPrompts.REFINE_QUERY_SYSTEM

        self.dataset_logger.log_refinement(
            system_prompt=system_instruction,
            user_input=query,
            refined_output=refined_query
        )
        return refined_query

    def parse_search_query(self, query: str, chat_history: List[Dict] = None) -> Dict:
        """Parse query into filters and semantic intent"""

        history_text = "No previous context."
        if chat_history:
            # unique last 3 pairs to keep context relevant but short
            relevant_history = chat_history[-6:]
            history_text = "\\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in relevant_history])

        prompt = AgentPrompts.STRUCTURED_SEARCH_PROMPT.format(query=query, history=history_text)
        response = self.llm.invoke([("human", prompt)])
        content = response.content.strip()

        # Clean markdown code blocks if present
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            content = match.group(1)

        result = {"filters": {}, "semantic_query": "", "limit": 5}
        try:
            import json
            parsed = json.loads(content)
            result["filters"] = parsed.get("filters", {})
            result["semantic_query"] = parsed.get("semantic_query", "")
            result["limit"] = int(parsed.get("limit", 5))
        except Exception:
            # Fallback if JSON fails - assume it's just a semantic query if simple text, or empty
            pass

        # Log for refinement fine-tuning
        self.dataset_logger.log_refinement(
            system_prompt=AgentPrompts.STRUCTURED_SEARCH_SYSTEM,
            user_input=query,
            refined_output=content
        )
        return result

    def resolve_target_date(self, query: str, context_date: str) -> str:
        """Resolve the target date based on user query and context"""
        prompt = AgentPrompts.RESOLVE_DATE.format(context_date=context_date, query=query)
        response = self.llm.invoke([("human", prompt)])
        return response.content.strip()
