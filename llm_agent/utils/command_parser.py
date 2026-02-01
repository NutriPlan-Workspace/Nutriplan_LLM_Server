from typing import List, Dict
import json
from llm_agent.utils.logger import logger

class CommandParser:
    """Helper to extract JSON commands from LLM text responses"""

    @staticmethod
    def extract_commands(response: str) -> List[Dict]:
        """Extract frontend commands from response using brace counting"""
        commands = []
        try:
            # Simple brace counting to find JSON objects
            idx = 0
            while idx < len(response):
                start = response.find('{', idx)
                if start == -1:
                    break

                # Check if this looks like our command type
                # Optimization: Look ahead for "FRONTEND_COMMAND" before parsing
                # But careful not to miss it if it's deeper.
                # For safety, let's just parse all top-level JSON objects we find.

                open_braces = 0
                end = start
                found = False

                for i in range(start, len(response)):
                    if response[i] == '{':
                        open_braces += 1
                    elif response[i] == '}':
                        open_braces -= 1

                    if open_braces == 0:
                        end = i + 1
                        found = True
                        break

                if found:
                    json_str = response[start:end]
                    try:
                        # Attempt parse
                        obj = json.loads(json_str)
                        if isinstance(obj, dict) and obj.get("type") in ["FRONTEND_COMMAND", "replace_food", "add_to_grocery", "search_food"]:
                            commands.append(obj)
                    except:
                        pass # Not valid JSON, ignore

                    idx = end
                else:
                    # Unbalanced or incomplete
                    idx = start + 1

        except Exception as e:
            logger.error(f"Error extracting commands: {e}")

        return commands
