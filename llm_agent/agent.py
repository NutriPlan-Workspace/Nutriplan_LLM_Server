from typing import List, Dict
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from llm_agent.rag.food_rag import FoodRAGPipeline
from llm_agent.rag.manual_rag import UserManualRAGPipeline
from llm_agent.tools.backend import BackendDataTool
from llm_agent.tools.web_search import WebSearchTool
from llm_agent.tools.web_search import WebSearchTool
from llm_agent.utils.logger import logger
from llm_agent.utils.dataset_logger import DatasetLogger
from llm_agent.utils.messages import LogMessages
from llm_agent.utils.prompts import AgentPrompts

# New modules
from llm_agent.utils.llm_reasoning import LLMReasoning
from llm_agent.utils.command_parser import CommandParser

class MealPlannerAgent:
    """LangChain Agent integrating Food RAG + User Manual RAG + Web Search"""

    def __init__(
        self,
        food_rag: FoodRAGPipeline,
        manual_rag: UserManualRAGPipeline,
        vllm_base_url: str,
        vllm_model_name: str,
        vllm_api_key: str,
        vllm_temperature: float,
        backend_url: str,
    ):
        logger.info(LogMessages.AGENT_INIT)
        # Initialize Dataset Logger -> Finetune LLM
        self.dataset_logger = DatasetLogger()

        # Tools in system
        self.food_rag = food_rag
        self.manual_rag = manual_rag
        self.backend_tool = BackendDataTool(base_url=backend_url)
        self.web_search_tool = WebSearchTool()

        self.llm = ChatOpenAI(
            base_url=vllm_base_url,
            api_key=vllm_api_key,
            model=vllm_model_name,
            temperature=vllm_temperature,
        )

        # Initialize Cognition Module
        self.llm_reasoning = LLMReasoning(self.llm)

        logger.info(LogMessages.AGENT_READY)

    def chat(self, message: str, chat_history: List[Dict] = None, auth_token: str = None):
        """Process user message with Streaming"""

        chat_history = chat_history or []

        logger.info("="*60)
        logger.info(LogMessages.AGENT_NEW_REQUEST.format(message=message))
        logger.info(f"[LLM Server] Chat History Length: {len(chat_history)}")
        logger.debug(f"[LLM Server] Auth Token: {'Present (' + auth_token[:10] + '...)' if auth_token else 'MISSING'}")
        logger.info("="*60)

        # 0. Summarize History
        if len(chat_history) > 5:
             logger.info(LogMessages.AGENT_SUMMARIZING_HISTORY)
             yield {"status": "thinking", "message": "Updating conversation memory..."}
             chat_history = self.llm_reasoning.summarize_history(chat_history)
             logger.info(LogMessages.AGENT_HISTORY_SUMMARIZED)

        # 1. Classify
        logger.info(LogMessages.AGENT_CLASSIFICATION_START)
        yield {"status": "thinking", "message": "Understanding your request..."}
        category = self.llm_reasoning.classify_query(message)
        logger.info(LogMessages.AGENT_CLASSIFICATION_RESULT.format(category=category))

        context_data = ""

        # --- PATH 1: FOOD SEARCH (Unified Hybrid: Semantic + Structured) ---
        if "ARITHMETIC" in category or "SEMANTIC" in category:
            logger.info(LogMessages.AGENT_PATH_SEMANTIC) # Using generic log message
            yield {"status": "thinking", "message": "Analyzing food criteria..."}

            # 1. Parse Query (Extract Filters + Semantic Intent)
            search_params = self.llm_reasoning.parse_search_query(message, chat_history)
            mongo_filters = search_params.get("filters")
            semantic_query = search_params.get("semantic_query")
            limit_k = search_params.get("limit", 5)

            logger.debug(f"[Agent] Filters: {mongo_filters}")
            logger.debug(f"[Agent] Semantic Query: '{semantic_query}'")
            logger.debug(f"[Agent] Limit: {limit_k}")

            # 2. Execute Hybrid Search
            yield {"status": "thinking", "message": "Searching food database..."}
            results = self.food_rag.search(query=semantic_query, k=limit_k, filters=mongo_filters)

            context_data = "\\n".join([doc.page_content for doc in results])
            logger.debug(LogMessages.AGENT_RAG_RESULTS.format(count=len(results), result=context_data))

        # --- PATH 3: PERSONAL_DATA (Backend) ---
        elif "PERSONAL_DATA" in category and auth_token:
            logger.info(LogMessages.AGENT_PATH_PERSONAL_DATA)
            yield {"status": "thinking", "message": "Checking your personal data..."}
            lower_msg = message.lower()
            logger.debug(f"[LLM Server] Lower message: {lower_msg}")

            # IMPORTANT: Check MEAL_PLAN keywords BEFORE "có gì" (which is ambiguous)
            if "plan" in lower_msg or "thực đơn" in lower_msg or "ăn gì" in lower_msg or "dinner" in lower_msg or "lunch" in lower_msg or "breakfast" in lower_msg or "meal" in lower_msg:
                logger.info(LogMessages.AGENT_BRANCH_MEAL_PLAN)

                # Default to today
                context_date = datetime.now().strftime("%Y-%m-%d")

                # Simply extract the context date first (as a fallback/reference)
                import re
                date_match = re.search(r"Date context: (\d{4}-\d{2}-\d{2})", message)
                if date_match:
                    context_date = date_match.group(1)

                # Use LLM to resolve the ACTUAL target date (handling user overrides like "tomorrow", "21/01")
                # Clean the message to remove "Date context" header so LLM doesn't get confused
                clean_query = message
                if "Request:" in message:
                    clean_query = message.split("Request:", 1)[1].strip()

                target_date = self.llm_reasoning.resolve_target_date(clean_query, context_date)
                logger.debug(LogMessages.AGENT_RESOLVED_DATE.format(target_date=target_date, context_date=context_date))

                logger.info(LogMessages.AGENT_FETCHING_MEAL_PLAN.format(date=target_date))
                yield {"status": "thinking", "message": f"Fetching meal plan for {target_date}..."}
                context_data = self.backend_tool.get_daily_plan(auth_token, target_date)
                logger.debug(LogMessages.AGENT_MEAL_PLAN_RESULT.format(result=context_data))
            elif "pantry" in lower_msg or "tủ" in lower_msg:
                logger.info(LogMessages.AGENT_BRANCH_PANTRY)
                yield {"status": "thinking", "message": "Checking your pantry..."}
                context_data = self.backend_tool.get_pantry_items(auth_token)
                logger.debug(LogMessages.AGENT_PANTRY_RESULT.format(result=context_data))
            else:
                logger.info(LogMessages.AGENT_BRANCH_USER_PROFILE)
                yield {"status": "thinking", "message": "Fetching your profile..."}
                context_data = self.backend_tool.get_user_profile(auth_token)
                logger.debug(LogMessages.AGENT_PROFILE_RESULT.format(result=context_data))

        # --- PATH 3b: PERSONAL_DATA without auth ---
        elif "PERSONAL_DATA" in category and not auth_token:
            logger.info(LogMessages.AGENT_PATH_PERSONAL_DATA_NO_AUTH)
            yield {"status": "thinking", "message": "Checking authentication..."}
            context_data = "LOGIN_REQUIRED: User is not logged in. They need to log in to access personal data like meal plans, pantry items, or profile information."
            logger.info("PERSONAL_DATA requested without auth token")

        # --- PATH 4: GENERAL / MANUAL ---
        elif "GENERAL" in category:
            logger.info(LogMessages.AGENT_PATH_GENERAL)
            if "how" in message.lower() or "làm sao" in message.lower() or "cách" in message.lower():
                yield {"status": "thinking", "message": "Searching user manual..."}
                context_data = self.manual_rag.semantic_search(message)
                if isinstance(context_data, list):
                    context_data = "\\n".join([r.get('text', '') for r in context_data])
                logger.debug(LogMessages.AGENT_MANUAL_SEARCH_RESULT.format(result=context_data[:200]))

        # --- PATH 5: WEB SEARCH ---
        elif "WEB_SEARCH" in category:
            logger.info(LogMessages.AGENT_PATH_WEB_SEARCH)
            yield {"status": "thinking", "message": "Searching the web..."}
            context_data = self.web_search_tool.search(message)
            logger.debug(LogMessages.AGENT_WEB_SEARCH_RESULT.format(result=context_data[:200]))

        # --- PATH 6: FRONTEND ACTION ---
        elif "FRONTEND_ACTION" in category:
            logger.info(LogMessages.AGENT_PATH_FRONTEND_ACTION)
            logger.debug(f"[LLM Server] -> auth_token received: {'Yes (' + auth_token[:10] + '...)' if auth_token else 'NO'}")
            # Navigation to public pages (login, register) doesn't require auth
            # But actions like add_to_grocery, swap_food do require auth
            lower_msg = message.lower()
            is_navigation_only = any(kw in lower_msg for kw in ["navigate", "go to", "đến trang", "chuyển", "mở"])
            is_public_page = any(page in lower_msg for page in ["login", "register", "đăng ký", "đăng nhập"])
            logger.debug(f"[LLM Server] -> is_navigation_only: {is_navigation_only}, is_public_page: {is_public_page}")

            if not auth_token and not (is_navigation_only and is_public_page):
                logger.warning("[LLM Server] -> FRONTEND_ACTION without auth (requires login)")
                yield {"status": "thinking", "message": "Checking authentication..."}
                context_data = "LOGIN_REQUIRED: User is not logged in. They need to log in to perform actions like swapping foods or adding items to grocery list. Navigation is done via markdown links."
                logger.info("FRONTEND_ACTION requested without auth token")

            else:
                yield {"status": "thinking", "message": "Processing action..."}
                # For actions, we often don't need external context, but we might need the meal plan if it's a swap.
                # However, usually the user instruction is enough for the LLM to generate the command.
                # We can optionally fetch the meal plan if "swap" is mentioned to confirm validity,
                # but to keep it simple and fast, we'll let the FE handle validation or user can confirm.
                context_data = "User wants to perform a UI action. Generate the corresponding FRONTEND_COMMAND if needed (only for add_to_grocery or swap_food). For navigation, use markdown links instead."


        # --- FINAL RESPONSE GENERATION ---
        logger.info(LogMessages.AGENT_GENERATING_RESPONSE)
        logger.debug(LogMessages.AGENT_CONTEXT_LENGTH.format(length=len(context_data)))
        yield {"status": "thinking", "message": "Drafting response..."}

        final_prompt = AgentPrompts.FINAL_PROMPT.format(context_data=context_data, query=message)
        logger.debug(LogMessages.AGENT_FINAL_PROMPT.format(prompt=final_prompt[:500]))

        history = []
        if chat_history:
            for msg in chat_history:
                if msg["role"] == "user":
                    history.append(HumanMessage(content=msg["content"]))
                else:
                    history.append(AIMessage(content=msg["content"]))

        messages = [
            ("system", AgentPrompts.MAIN_SYSTEM),
            *history,
            ("human", final_prompt)
        ]

        # Enable streaming
        logger.info(LogMessages.AGENT_STARTING_STREAM)
        full_response = ""
        for chunk in self.llm.stream(messages):
            content = chunk.content
            if content:
                full_response += content
                yield {"status": "token", "content": content}

        commands = CommandParser.extract_commands(full_response)

        # Log the full turn for dataset
        # Log the full turn for dataset (Chat Fine-tuning)
        self.dataset_logger.log_generation(
            system_prompt=AgentPrompts.MAIN_SYSTEM,
            user_input=final_prompt, # This contains Context + Query
            assistant_response=full_response,
            context_data=context_data
        )

        yield {"status": "done", "commands": commands}

