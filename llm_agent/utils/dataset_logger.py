import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from llm_agent.utils.logger import logger

class DatasetLogger:
    """
    Logger designed to capture conversation data for LLM fine-tuning.
    Logs each turn as a JSON object in a JSONL file.
    """

    def __init__(self, refinement_log: str = "dataset/dataset_refinement.jsonl", generation_log: str = "dataset/dataset_generation.jsonl"):
        """
        Initialize the dataset logger.

        Args:
            refinement_log: Path for refinement step logs (System -> User -> Semantic Query).
            generation_log: Path for final generation logs (System -> User+Context -> Assistant).
        """
        # Ensure path is absolute if not provided
        if not os.path.isabs(refinement_log):
            refinement_log = os.path.join(os.getcwd(), refinement_log)
        if not os.path.isabs(generation_log):
            generation_log = os.path.join(os.getcwd(), generation_log)

        self.refinement_log = refinement_log
        self.generation_log = generation_log

        # Generate a session ID for this instance
        self.session_id = str(uuid.uuid4())
        self.turn_count = 0

        logger.info(f"[DatasetLogger] Refinement Log: {self.refinement_log}")
        logger.info(f"[DatasetLogger] Refinement Log: {self.refinement_log}")
        logger.info(f"[DatasetLogger] Generation Log: {self.generation_log}")
        logger.info(f"[DatasetLogger] Session ID: {self.session_id}")

    def log_refinement(self, system_prompt: str, user_input: str, refined_output: str):
        """
        Log the query refinement step (User Query -> Semantic Query / Filter).
        """
        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": refined_output}
            ],
            "metadata": {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat() + "Z",
                "type": "refinement"
            }
        }
        self._write_log(self.refinement_log, entry)

    def log_generation(
        self,
        system_prompt: str,
        user_input: str,
        assistant_response: str,
        context_data: str = None
    ):
        """
        Log the final chat turn (System + Context + User -> Assistant).
        Note: We inject context into the user message or system prompt depending on strategy.
        Here we follow the Agent's strategy: Context is part of the final prompt (User message).
        """
        self.turn_count += 1

        # In our agent, 'user_input' passed here is actually the Combined Prompt (Context + Query)
        # OR we can format it explicitly here if we passed raw inputs.
        # Let's assume the caller passes the FINAL PROMPT that went to the LLM as 'user_input'
        # OR robustly construct it.

        # If the caller passes the raw components, we can construct the structure:

        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_response}
            ],
            "metadata": {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat() + "Z",
                "turn_id": self.turn_count,
                "context_preview": context_data[:200] if context_data else "N/A"
            }
        }
        self._write_log(self.generation_log, entry)

    def _write_log(self, file_path: str, entry: Dict):
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"❌ [DatasetLogger] Failed to write log to {file_path}: {e}")

# Global instance can be created if needed, but Agent will likely own it.
